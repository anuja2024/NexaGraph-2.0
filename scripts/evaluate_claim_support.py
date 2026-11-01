import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_PATH = PROJECT_ROOT / "data" / "staging" / "siemens_pdf" / "claim_evaluation.json"

ANSWER_PATH = PROJECT_ROOT / "data" / "staging" / "siemens_pdf" / "grounded_answer.json"

SOURCE_ID_PATTERN = re.compile(r"\[(S\d+)\]")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def main() -> None:
    load_dotenv(ENV_PATH, override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    ).strip()

    if not api_key or api_key == "PASTE_YOUR_REAL_KEY_HERE":
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    if not ANSWER_PATH.is_file():
        print(
            "The saved answer file does not exist yet:\n"
            f"{ANSWER_PATH}\n\n"
            "This evaluator needs a saved answer and its exact "
            "source passages. We will create that evidence package "
            "in the next checkpoint."
        )
        print("No evaluation was run; no existing files were changed.")
        return

    with ANSWER_PATH.open("r", encoding="utf-8") as file:
        package = json.load(file)

    answer = package.get("answer", "")
    sources = package.get("sources", [])

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Evidence package has no valid answer")

    if not isinstance(sources, list) or not sources:
        raise ValueError("Evidence package has no source passages")

    source_map = {}
    for source in sources:
        source_id = source.get("source_id")
        text = source.get("text")
        filename = source.get("source_file")
        page = source.get("page_number")

        if (
            not isinstance(source_id, str)
            or not re.fullmatch(r"S\d+", source_id)
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(filename, str)
            or not filename
            or not isinstance(page, int)
            or page < 1
        ):
            raise ValueError(
                "A source is missing a valid ID, passage, filename, "
                "or page number"
            )

        if source_id in source_map:
            raise ValueError(f"Duplicate source ID: {source_id}")

        source_map[source_id] = source

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(answer.strip())
        if sentence.strip()
    ]

    claim_results = []
    missing_citations = []
    unknown_citations = []

    for sentence_number, sentence in enumerate(sentences, start=1):
        cited_ids = SOURCE_ID_PATTERN.findall(sentence)
        content = SOURCE_ID_PATTERN.sub("", sentence).strip()

        # Headings and empty fragments are not factual claims.
        if not content or content.endswith(":"):
            continue

        unknown_ids = sorted(set(cited_ids) - set(source_map))

        if not cited_ids:
            missing_citations.append(sentence_number)

        if unknown_ids:
            unknown_citations.append(
                {
                    "sentence_number": sentence_number,
                    "unknown_source_ids": unknown_ids,
                }
            )

        claim_results.append(
            {
                "sentence_number": sentence_number,
                "claim": content,
                "cited_source_ids": sorted(set(cited_ids)),
                "unknown_source_ids": unknown_ids,
                "has_citation": bool(cited_ids),
                "citation_ids_exist": bool(cited_ids) and not unknown_ids,
            }
        )

    print("Deterministic citation checks")
    print(f"Answer sentences examined: {len(claim_results)}")
    print(f"Sentences without citations: {missing_citations}")
    print(f"Unknown citation IDs: {unknown_citations}")

    if missing_citations or unknown_citations:
        print(
            "FAIL: Basic citation checks found issues. "
            "Do not treat the answer as verified."
        )
        raise SystemExit(1)

    evidence_blocks = []

    for source_id, source in source_map.items():
        evidence_blocks.append(
            f"[{source_id}] File: {source['source_file']}; "
            f"page: {source['page_number']}\n"
            f"Passage: {source['text']}"
        )

    assessment_prompt = f"""
You are evaluating evidence support for a retrieval-augmented
knowledge assistant. You are not writing a new answer.

Assess every numbered claim below using only the supplied evidence.
For each claim, return:
- claim_number
- verdict: "supported", "partially_supported", "unsupported",
  or "uncertain"
- supporting_source_ids: only source IDs whose passages support it
- explanation: concise explanation based on the supplied text

Do not assume a citation is correct just because its ID exists.
Do not use outside knowledge.
If a passage does not establish the claim, mark it unsupported
or uncertain. If the evidence only supports part of the claim,
mark it partially_supported.
Return only valid JSON in this format:
{{
  "assessments": [
    {{
      "claim_number": 1,
      "verdict": "supported",
      "supporting_source_ids": ["S1"],
      "explanation": "The passage states the claimed figure."
    }}
  ]
}}

Claims:
{json.dumps(
    [
        {
            "claim_number": item["sentence_number"],
            "claim": item["claim"],
            "cited_source_ids": item["cited_source_ids"],
        }
        for item in claim_results
    ],
    ensure_ascii=False,
    indent=2,
)}

Evidence:
{chr(10).join(evidence_blocks)}
""".strip()

    print("\nCalling Gemini to assess claim support...")
    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=assessment_prompt,
            config={
                "response_mime_type": "application/json",
            },
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    response_text = (response.text or "").strip()

    if not response_text:
        raise RuntimeError("Gemini returned an empty assessment")

    try:
        assessment_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON; assessment was not accepted"
        ) from exc

    assessments = assessment_data.get("assessments")

    if not isinstance(assessments, list):
        raise ValueError("Assessment JSON lacks an assessments list")

    expected_numbers = {
        item["sentence_number"] for item in claim_results
    }
    returned_numbers = [
        item.get("claim_number")
        for item in assessments
        if isinstance(item, dict)
    ]

    if (
        len(returned_numbers) != len(expected_numbers)
        or set(returned_numbers) != expected_numbers
        or len(returned_numbers) != len(set(returned_numbers))
    ):
        raise ValueError(
            "Assessment did not return exactly one result "
            "for every claim"
        )

    allowed_verdicts = {
        "supported",
        "partially_supported",
        "unsupported",
        "uncertain",
    }

    for assessment in assessments:
        if assessment.get("verdict") not in allowed_verdicts:
            raise ValueError("Assessment contains an invalid verdict")

        support_ids = assessment.get("supporting_source_ids")

        if not isinstance(support_ids, list):
            raise ValueError(
                "Assessment supporting_source_ids must be a list"
            )

        if any(
            not isinstance(source_id, str)
            or source_id not in source_map
            for source_id in support_ids
        ):
            raise ValueError(
                "Assessment referenced an unknown supporting source ID"
            )

    assessment_by_number = {
        item["claim_number"]: item
        for item in assessments
    }

    for claim in claim_results:
        claim["support_assessment"] = assessment_by_number[
            claim["sentence_number"]
        ]

    output = {
        "model": model_name,
        "evaluation_type": "citation_integrity_and_model_assessed_support",
        "answer": answer,
        "sources": sources,
        "claims": claim_results,
        "limitations": [
            "Model-assessed support is not definitive proof.",
            "Citation ID validity does not guarantee entailment.",
            "Independent review and a human-verified evaluation set "
            "are required before production use.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\nClaim support assessments")
    for claim in claim_results:
        assessment = claim["support_assessment"]
        print(
            f"Claim {claim['sentence_number']}: "
            f"{assessment['verdict']}"
        )
        print(f"  Claim: {claim['claim']}")
        print(
            "  Supporting sources: "
            f"{assessment['supporting_source_ids']}"
        )
        print(f"  Explanation: {assessment['explanation']}")

    print(f"\nEvaluation saved to: {OUTPUT_PATH}")
    print(
        "NOTE: This is a model-assisted evaluation. "
        "Review uncertain and unsupported claims manually."
    )


if __name__ == "__main__":
    main()

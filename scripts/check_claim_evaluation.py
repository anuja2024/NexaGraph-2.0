import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "siemens_pdf"
    / "claim_evaluation.json"
)

ALLOWED_VERDICTS = {
    "supported",
    "partially_supported",
    "unsupported",
    "uncertain",
}


def main() -> None:
    if not EVALUATION_PATH.is_file():
        raise FileNotFoundError(
            f"Evaluation file not found: {EVALUATION_PATH}"
        )

    with EVALUATION_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    claims = data.get("claims")
    sources = data.get("sources")

    if not isinstance(claims, list) or not claims:
        raise ValueError("Evaluation has no claim records")

    if not isinstance(sources, list) or not sources:
        raise ValueError("Evaluation has no source registry")

    source_map = {}

    for source in sources:
        source_id = source.get("source_id")

        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Source registry contains an invalid ID")

        if source_id in source_map:
            raise ValueError(f"Duplicate source ID: {source_id}")

        source_map[source_id] = source

    passed = 0
    failed = 0
    review_required = []

    def check(condition: bool, message: str) -> None:
        nonlocal passed, failed

        if condition:
            passed += 1
            print(f"PASS: {message}")
        else:
            failed += 1
            print(f"FAIL: {message}")

    for claim in claims:
        number = claim.get("sentence_number")
        cited_ids = claim.get("cited_source_ids")
        assessment = claim.get("support_assessment")

        if not isinstance(cited_ids, list):
            check(False, f"Claim {number}: citation list exists")
            continue

        check(
            bool(cited_ids),
            f"Claim {number}: at least one citation is attached",
        )

        check(
            all(source_id in source_map for source_id in cited_ids),
            f"Claim {number}: all cited IDs exist in the registry",
        )

        if not isinstance(assessment, dict):
            check(False, f"Claim {number}: support assessment exists")
            review_required.append(number)
            continue

        verdict = assessment.get("verdict")
        supporting_ids = assessment.get("supporting_source_ids")

        check(
            verdict in ALLOWED_VERDICTS,
            f"Claim {number}: assessment verdict is valid",
        )

        check(
            isinstance(supporting_ids, list),
            f"Claim {number}: supporting source list exists",
        )

        if not isinstance(supporting_ids, list):
            review_required.append(number)
            continue

        check(
            all(source_id in source_map for source_id in supporting_ids),
            f"Claim {number}: assessor used known source IDs",
        )

        cited_set = set(cited_ids)
        supporting_set = set(supporting_ids)

        check(
            bool(cited_set.intersection(supporting_set)),
            f"Claim {number}: at least one cited source is assessed as supporting",
        )

        if verdict != "supported":
            review_required.append(number)

        if not supporting_set.issubset(cited_set):
            review_required.append(number)
            print(
                f"REVIEW: Claim {number} has assessor support IDs "
                "that are not all cited in the answer."
            )

    print("\n" + "=" * 72)
    print(f"Checks passed: {passed}")
    print(f"Checks failed: {failed}")
    print(
        "Claims requiring review: "
        f"{sorted(set(review_required))}"
    )

    if failed:
        print("CITATION CONSISTENCY CHECK FAILED")
        raise SystemExit(1)

    if review_required:
        print(
            "STRUCTURAL CHECK PASSED WITH CLAIMS REQUIRING REVIEW"
        )
    else:
        print("CITATION CONSISTENCY CHECK PASSED")

    print(
        "This script checks citation consistency, not factual truth. "
        "Human-verified evidence evaluation is still required."
    )


if __name__ == "__main__":
    main()

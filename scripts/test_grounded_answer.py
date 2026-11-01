import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_hybrid import (  # noqa: E402
    CANDIDATE_LIMIT,
    MODEL_NAME,
    QDRANT_PATH,
    bm25_search,
    load_chunks,
    reciprocal_rank_fusion,
    semantic_search,
)

ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "siemens_pdf"
    / "grounded_answer.json"
)

PROJECT_ID = "corporate"
QUERY = "What were Siemens' revenue and financial performance in fiscal 2025?"
EVIDENCE_LIMIT = 6
SOURCE_ID_PATTERN = re.compile(r"\[(S\d+)\]")


def main() -> None:
    load_dotenv(ENV_PATH, override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    ).strip()

    if not api_key or api_key == "PASTE_YOUR_REAL_KEY_HERE":
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    if not model_name:
        raise RuntimeError("GEMINI_MODEL is empty")

    print("Loading Siemens corpus...")
    all_chunks = load_chunks()

    project_chunks = [
        chunk
        for chunk in all_chunks
        if chunk.get("project_id") == PROJECT_ID
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]

    if not project_chunks:
        raise RuntimeError(f"No chunks found for {PROJECT_ID}")

    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        print("Retrieving evidence using BM25 and Qdrant...")

        bm25_results = bm25_search(
            project_chunks,
            QUERY,
            CANDIDATE_LIMIT,
        )

        embedding_model = TextEmbedding(model_name=MODEL_NAME)

        semantic_results = semantic_search(
            client,
            embedding_model,
            PROJECT_ID,
            QUERY,
            CANDIDATE_LIMIT,
        )

        fused_results = reciprocal_rank_fusion(
            [bm25_results, semantic_results]
        )

        sources = []
        seen = set()

        for item in fused_results:
            record = item["record"]

            if record.get("project_id") != PROJECT_ID:
                continue

            source_file = record.get("source_file")
            page_number = record.get("page_number")
            document_id = record.get("document_id")
            text = record.get("text")

            if not (
                isinstance(source_file, str)
                and source_file
                and isinstance(page_number, int)
                and page_number >= 1
                and isinstance(document_id, str)
                and document_id
                and isinstance(text, str)
                and text.strip()
            ):
                continue

            key = (document_id, source_file, page_number, text)

            if key in seen:
                continue

            seen.add(key)
            sources.append({
                "source_id": f"S{len(sources) + 1}",
                "project_id": PROJECT_ID,
                "document_id": document_id,
                "source_file": source_file,
                "page_number": page_number,
                "text": text.strip(),
            })

            if len(sources) >= EVIDENCE_LIMIT:
                break

        if not sources:
            raise RuntimeError("No valid evidence passages retrieved")

        print(f"Evidence passages selected: {len(sources)}")

        evidence_text = "\n\n".join(
            f"[{source['source_id']}]\n"
            f"Document: {source['source_file']}\n"
            f"Page: {source['page_number']}\n"
            f"Passage:\n{source['text']}"
            for source in sources
        )

        prompt = f"""
You are NexaGraph, an evidence-grounded company knowledge assistant.

Question:
{QUERY}

Answer using only the evidence passages below.

Rules:
- Do not use outside knowledge or guess missing facts.
- Cite every factual sentence with source IDs such as [S1].
- Use only source IDs provided below.
- Never invent a filename, page, figure, or citation.
- If evidence is insufficient, say so explicitly.
- Distinguish reported figures from interpretation.
- Be concise.

Evidence:
{evidence_text}
""".strip()

        print(f"Calling Gemini model: {model_name}")
        gemini = genai.Client(api_key=api_key)

        try:
            response = gemini.models.generate_content(
                model=model_name,
                contents=prompt,
            )
        finally:
            close = getattr(gemini, "close", None)
            if callable(close):
                close()

        answer = (response.text or "").strip()

        if not answer:
            raise RuntimeError("Gemini returned an empty answer")

        valid_ids = {source["source_id"] for source in sources}
        cited_ids = SOURCE_ID_PATTERN.findall(answer)
        unknown_ids = sorted(set(cited_ids) - valid_ids)

        print("\n" + "=" * 72)
        print("GEMINI ANSWER")
        print("=" * 72)
        print(answer)

        print("\nBACKEND SOURCE REGISTRY")
        for source in sources:
            print(
                f"[{source['source_id']}] "
                f"{source['source_file']} "
                f"(page {source['page_number']})"
            )

        if not cited_ids:
            raise RuntimeError("Answer contains no source citations")

        if unknown_ids:
            raise RuntimeError(
                f"Answer contains unknown source IDs: {unknown_ids}"
            )

        package = {
            "project_id": PROJECT_ID,
            "query": QUERY,
            "model": model_name,
            "answer": answer,
            "sources": sources,
            "citation_integrity": {
                "cited_source_ids": sorted(set(cited_ids)),
                "unknown_source_ids": unknown_ids,
                "all_cited_ids_exist": True,
            },
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(
                    package,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            temporary_path.replace(OUTPUT_PATH)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        print(f"\nPASS: Citation IDs map to retrieved sources")
        print(f"PASS: Evidence package saved to: {OUTPUT_PATH}")
        print(
            "NOTE: This confirms source-ID integrity only, "
            "not factual support for every claim."
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()

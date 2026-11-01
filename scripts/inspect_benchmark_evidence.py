import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = (
    ROOT / "data" / "staging" / "siemens_pdf" / "chunks.jsonl"
)

TARGETS = [
    {
        "question": "What were Siemens' revenue and financial performance in fiscal 2025?",
        "project_id": "corporate",
        "pages": [22, 25],
    },
    {
        "question": "How did Siemens' business segments perform?",
        "project_id": "corporate",
        "pages": [5, 81, 82],
    },
    {
        "question": "How does Siemens address carbon emissions?",
        "project_id": "sustainability",
        "pages": [25, 28, 30, 31],
    },
    {
        "question": "What decarbonization measures does Siemens report?",
        "project_id": "sustainability",
        "pages": [25, 30, 31],
    },
    {
        "question": "How can industrial automation systems be protected against cyber threats?",
        "project_id": "industrial_automation",
        "pages": [5, 9, 13, 17],
    },
    {
        "question": "What security measures are recommended for industrial networks?",
        "project_id": "industrial_automation",
        "pages": [9, 13, 17],
    },
]


def main() -> None:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Corpus not found: {CHUNKS_PATH}")

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print(f"Corpus chunks loaded: {len(chunks)}")

    for case_number, case in enumerate(TARGETS, start=1):
        print("\n" + "=" * 88)
        print(f"CASE {case_number}: {case['question']}")
        print(f"Project: {case['project_id']}")

        matches = [
            chunk for chunk in chunks
            if chunk.get("project_id") == case["project_id"]
            and chunk.get("page_number") in case["pages"]
        ]

        matches.sort(
            key=lambda chunk: (
                chunk.get("page_number", 0),
                chunk.get("chunk_index", 0),
            )
        )

        if not matches:
            print("NO MATCHING CHUNKS")
            continue

        for chunk in matches:
            print("-" * 72)
            print(f"Document ID: {chunk.get('document_id')}")
            print(f"File: {chunk.get('source_file')}")
            print(f"Page: {chunk.get('page_number')}")
            print(f"Chunk index: {chunk.get('chunk_index')}")
            print("Passage:")
            print(chunk.get("text", ""))


if __name__ == "__main__":
    main()

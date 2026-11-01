import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "staging" / "siemens_pdf" / "chunks.jsonl"
OUTPUT = ROOT / "data" / "staging" / "siemens_pdf" / "retrieval_benchmark.json"


CASES = [
    {
        "case_id": "corporate_financial_performance",
        "project_id": "corporate",
        "question": "What were Siemens' revenue and financial performance in fiscal 2025?",
        "expected_evidence": [
            {"page_number": 22, "chunk_index": 2},
            {"page_number": 22, "chunk_index": 3},
            {"page_number": 22, "chunk_index": 7},
        ],
        "status": "candidate_evidence_verified_from_inspected_passages",
        "evaluation_eligible": True,
    },
    {
        "case_id": "corporate_segment_performance",
        "project_id": "corporate",
        "question": "How did Siemens' business segments perform?",
        "expected_evidence": [
            {"page_number": 22, "chunk_index": 0},
        ],
        "status": "needs_more_segment_specific_evidence",
        "evaluation_eligible": False,
    },
    {
        "case_id": "sustainability_carbon_emissions",
        "project_id": "sustainability",
        "question": "How does Siemens address carbon emissions?",
        "expected_evidence": [
            {"page_number": 30, "chunk_index": 4},
            {"page_number": 31, "chunk_index": 3},
            {"page_number": 31, "chunk_index": 5},
        ],
        "status": "candidate_evidence_verified_from_inspected_passages",
        "evaluation_eligible": True,
    },
    {
        "case_id": "sustainability_decarbonization",
        "project_id": "sustainability",
        "question": "What decarbonization measures does Siemens report?",
        "expected_evidence": [
            {"page_number": 25, "chunk_index": 5},
            {"page_number": 30, "chunk_index": 4},
            {"page_number": 31, "chunk_index": 3},
        ],
        "status": "candidate_evidence_verified_from_inspected_passages",
        "evaluation_eligible": True,
    },
    {
        "case_id": "industrial_cybersecurity_protection",
        "project_id": "industrial_automation",
        "question": "How can industrial automation systems be protected against cyber threats?",
        "expected_evidence": [
            {"page_number": 9, "chunk_index": 0},
            {"page_number": 9, "chunk_index": 1},
            {"page_number": 17, "chunk_index": 0},
        ],
        "status": "candidate_evidence_verified_from_inspected_passages",
        "evaluation_eligible": True,
    },
    {
        "case_id": "industrial_network_security",
        "project_id": "industrial_automation",
        "question": "What security measures are recommended for industrial networks?",
        "expected_evidence": [
            {"page_number": 9, "chunk_index": 0},
            {"page_number": 17, "chunk_index": 0},
        ],
        "status": "candidate_evidence_verified_from_inspected_passages",
        "evaluation_eligible": True,
    },
]


def main():
    if not CORPUS.is_file():
        raise FileNotFoundError(f"Corpus not found: {CORPUS}")

    chunks = {}
    with CORPUS.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on corpus line {line_number}"
                ) from exc

            key = (
                item.get("project_id"),
                item.get("page_number"),
                item.get("chunk_index"),
            )

            if key in chunks:
                raise ValueError(f"Duplicate corpus chunk key: {key}")

            chunks[key] = item

    benchmark = {
        "benchmark_version": 2,
        "corpus": "Siemens public-document staging corpus",
        "evidence_matching": "exact_chunk_identity",
        "cases": [],
    }

    errors = []

    for case in CASES:
        resolved = []

        for expected in case["expected_evidence"]:
            key = (
                case["project_id"],
                expected["page_number"],
                expected["chunk_index"],
            )
            chunk = chunks.get(key)

            if chunk is None:
                errors.append(
                    f"{case['case_id']}: missing corpus chunk {key}"
                )
                continue

            resolved.append({
                "document_id": chunk["document_id"],
                "project_id": chunk["project_id"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            })

        record = dict(case)
        record["expected_evidence"] = resolved
        benchmark["cases"].append(record)

    if errors:
        print("BENCHMARK VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    OUTPUT.write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Benchmark version: {benchmark['benchmark_version']}")
    print(f"Cases created: {len(benchmark['cases'])}")
    print(
        "Evidence references resolved: "
        f"{sum(len(case['expected_evidence']) for case in benchmark['cases'])}"
    )
    print(
        "Evaluation-eligible cases: "
        f"{sum(case['evaluation_eligible'] for case in benchmark['cases'])}"
    )

    for case in benchmark["cases"]:
        print(
            f"{case['case_id']}: "
            f"evidence={len(case['expected_evidence'])}, "
            f"eligible={case['evaluation_eligible']}, "
            f"status={case['status']}"
        )

    print(f"Output: {OUTPUT}")
    print("PASS: all evidence references resolve to corpus chunks.")
    print("NOTE: resolving a reference does not independently prove relevance.")


if __name__ == "__main__":
    main()

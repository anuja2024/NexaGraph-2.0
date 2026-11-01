
import json
import math
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastembed import TextEmbedding
from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "staging" / "siemens_pdf"
CORPUS_PATH = DATA / "chunks.jsonl"
BENCHMARK_PATH = DATA / "retrieval_benchmark.json"
REPORT_PATH = DATA / "candidate_depth_experiment.json"
QDRANT_PATH = ROOT / "data" / "vector_store" / "qdrant_local"

COLLECTION = "nexagraph_siemens_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

TOP_K = 5
CANDIDATE_DEPTHS = (10, 20)
RRF_K = 60
MAX_SEQUENCE_LENGTH = 256


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def record_key(record):
    return (
        record.get("project_id"),
        record.get("document_id"),
        record.get("source_file"),
        record.get("page_number"),
        record.get("chunk_index"),
        record.get("text", ""),
    )


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def eligible(case):
    status = str(case.get("status", "")).lower()
    return (
        "needs_more" not in status
        and case.get("evaluation_eligible", True) is True
    )


def validate_benchmark(corpus, benchmark):
    corpus_keys = {record_key(item) for item in corpus}

    for case in benchmark.get("cases", []):
        expected = case.get("expected_evidence", [])
        if not expected:
            raise ValueError(
                f"No expected evidence for {case.get('case_id')}"
            )

        for item in expected:
            if record_key(item) not in corpus_keys:
                raise ValueError(
                    f"Benchmark evidence missing from corpus: "
                    f"{case.get('case_id')} "
                    f"page={item.get('page_number')} "
                    f"chunk={item.get('chunk_index')}"
                )

            if item.get("project_id") != case.get("project_id"):
                raise ValueError(
                    f"Cross-project evidence in {case.get('case_id')}"
                )


def bm25_search(chunks, query, limit):
    tokenized = []
    documents = []

    for chunk in chunks:
        terms = tokenize(chunk.get("text", ""))
        if terms:
            documents.append(chunk)
            tokenized.append(terms)

    query_terms = tokenize(query)
    if not documents or not query_terms:
        return []

    model = BM25Okapi(tokenized)
    scores = model.get_scores(query_terms)
    query_term_set = set(query_terms)
    ranked = []

    for index, score in enumerate(scores):
        if not query_term_set.intersection(tokenized[index]):
            continue

        record = dict(documents[index])
        record["_bm25_score"] = float(score)
        ranked.append(record)

    ranked.sort(
        key=lambda item: item["_bm25_score"],
        reverse=True,
    )
    return ranked[:limit]


def semantic_search(client, embedder, project_id, query, limit):
    vectors = list(embedder.query_embed(query))
    if len(vectors) != 1:
        raise RuntimeError("Expected exactly one query embedding")

    result = client.query_points(
        collection_name=COLLECTION,
        query=[float(value) for value in vectors[0]],
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_id),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )

    return [
        dict(point.payload)
        for point in result.points
        if point.payload
    ]


def reciprocal_rank_fusion(rankings):
    fused = {}

    for ranking in rankings:
        for rank, record in enumerate(ranking, start=1):
            identity = record_key(record)

            if identity not in fused:
                fused[identity] = {
                    "record": record,
                    "score": 0.0,
                }

            fused[identity]["score"] += 1.0 / (RRF_K + rank)

    values = list(fused.values())
    values.sort(key=lambda item: item["score"], reverse=True)
    return [item["record"] for item in values]


def load_reranker():
    model_path = hf_hub_download(
        repo_id=RERANKER_MODEL,
        filename="onnx/model.onnx",
    )
    tokenizer_path = hf_hub_download(
        repo_id=RERANKER_MODEL,
        filename="tokenizer.json",
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    return tokenizer, session


def rerank(tokenizer, session, query, candidates):
    input_names = {item.name for item in session.get_inputs()}
    scored = []

    for record in candidates:
        encoded = tokenizer.encode(query, record.get("text", ""))

        inputs = {
            "input_ids": np.asarray([encoded.ids], dtype=np.int64),
            "attention_mask": np.asarray(
                [encoded.attention_mask], dtype=np.int64
            ),
        }

        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.asarray(
                [encoded.type_ids], dtype=np.int64
            )

        output = session.run(None, inputs)[0]
        score = float(np.asarray(output).reshape(-1)[0])
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored]


def calculate_metrics(ranked, expected):
    expected_keys = {record_key(item) for item in expected}
    if not expected_keys:
        raise ValueError("Expected evidence cannot be empty")

    top = ranked[:TOP_K]
    retrieved_keys = [record_key(item) for item in top]
    hits = [1 if key in expected_keys else 0 for key in retrieved_keys]

    found = set(retrieved_keys).intersection(expected_keys)
    recall = len(found) / len(expected_keys)

    mrr = next(
        (
            1.0 / rank
            for rank, hit in enumerate(hits, start=1)
            if hit
        ),
        0.0,
    )

    dcg = sum(
        hit / math.log2(rank + 1)
        for rank, hit in enumerate(hits, start=1)
    )
    ideal_count = min(len(expected_keys), TOP_K)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return {
        "recall_at_5": recall,
        "mrr": mrr,
        "ndcg_at_5": dcg / ideal_dcg if ideal_dcg else 0.0,
        "retrieved_evidence_count": len(found),
        "expected_evidence_count": len(expected_keys),
        "top_5": [rank_entry(item) for item in top],
    }


def rank_entry(record):
    return {
        "source_file": record.get("source_file"),
        "page_number": record.get("page_number"),
        "chunk_index": record.get("chunk_index"),
    }


def evidence_rankings(ranking, expected):
    positions = {
        record_key(record): rank
        for rank, record in enumerate(ranking, start=1)
    }

    result = []
    for item in expected:
        result.append({
            **rank_entry(item),
            "rank": positions.get(record_key(item)),
            "found_within_candidate_depth": (
                record_key(item) in positions
            ),
        })
    return result


def main():
    for path in (CORPUS_PATH, BENCHMARK_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    corpus = load_jsonl(CORPUS_PATH)
    benchmark = json.loads(
        BENCHMARK_PATH.read_text(encoding="utf-8")
    )
    validate_benchmark(corpus, benchmark)

    cases = benchmark["cases"]
    eligible_cases = [case for case in cases if eligible(case)]
    excluded_cases = [case for case in cases if not eligible(case)]

    if not eligible_cases:
        raise ValueError("No benchmark cases are eligible")

    print(f"Corpus records: {len(corpus)}")
    print(f"Benchmark cases: {len(cases)}")
    print(f"Eligible cases: {len(eligible_cases)}")
    print(f"Diagnostic-only cases: {len(excluded_cases)}")
    print(f"Candidate depths: {CANDIDATE_DEPTHS}")

    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    client = QdrantClient(path=str(QDRANT_PATH))

    report = {
        "benchmark_version": benchmark.get("benchmark_version"),
        "candidate_depths": list(CANDIDATE_DEPTHS),
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "eligible_case_ids": [
            case["case_id"] for case in eligible_cases
        ],
        "diagnostic_only_case_ids": [
            case["case_id"] for case in excluded_cases
        ],
        "experiments": {},
        "limitations": [
            "Metrics use the existing candidate evidence labels.",
            "Evidence labels may be incomplete or overly narrow.",
            "Retrieval of labeled chunks does not prove answer factuality.",
            "The cross-encoder reranks only the fused candidate pool.",
            "Changing candidate depth also changes the pool available to fusion.",
        ],
    }

    try:
        tokenizer, session = load_reranker()

        for depth in CANDIDATE_DEPTHS:
            print("\n" + "=" * 72)
            print(f"CANDIDATE DEPTH: {depth}")

            method_totals = {
                method: {
                    "recall_at_5": [],
                    "mrr": [],
                    "ndcg_at_5": [],
                }
                for method in (
                    "bm25",
                    "semantic",
                    "hybrid_rrf",
                    "cross_encoder",
                )
            }
            case_results = []

            for case in cases:
                project_id = case["project_id"]
                query = case["question"]

                project_chunks = [
                    item for item in corpus
                    if item.get("project_id") == project_id
                    and isinstance(item.get("text"), str)
                    and item["text"].strip()
                ]

                bm25 = bm25_search(project_chunks, query, depth)
                semantic = semantic_search(
                    client, embedder, project_id, query, depth
                )
                hybrid = reciprocal_rank_fusion([bm25, semantic])
                cross_encoder = rerank(
                    tokenizer, session, query, hybrid[:depth]
                )

                rankings = {
                    "bm25": bm25,
                    "semantic": semantic,
                    "hybrid_rrf": hybrid,
                    "cross_encoder": cross_encoder,
                }

                case_eligible = eligible(case)
                case_result = {
                    "case_id": case["case_id"],
                    "included_in_aggregate": case_eligible,
                    "methods": {},
                }

                print(
                    f"\n{case['case_id']} "
                    f"({'eligible' if case_eligible else 'diagnostic only'})"
                )

                for method, ranking in rankings.items():
                    result = calculate_metrics(
                        ranking, case["expected_evidence"]
                    )

                    # Save top five metrics plus the full diagnostic ranking.
                    result["top_candidates"] = [
                        rank_entry(item) for item in ranking[:20]
                    ]
                    result["expected_evidence_ranks"] = evidence_rankings(
                        ranking, case["expected_evidence"]
                    )
                    case_result["methods"][method] = result

                    if case_eligible:
                        for metric in method_totals[method]:
                            method_totals[method][metric].append(
                                result[metric]
                            )

                    print(
                        f"  {method:14s} "
                        f"Recall@5={result['recall_at_5']:.3f} "
                        f"MRR={result['mrr']:.3f} "
                        f"nDCG@5={result['ndcg_at_5']:.3f}"
                    )

                if case["case_id"] == "corporate_financial_performance":
                    print("\n  FINANCIAL EVIDENCE RANKS:")
                    for method, ranking in rankings.items():
                        print(f"    {method}:")
                        for item in evidence_rankings(
                            ranking, case["expected_evidence"]
                        ):
                            print(
                                f"      page={item['page_number']} "
                                f"chunk={item['chunk_index']} "
                                f"rank={item['rank']}"
                            )

                case_results.append(case_result)

            aggregate = {
                method: {
                    metric: (
                        sum(values) / len(values) if values else 0.0
                    )
                    for metric, values in metrics.items()
                }
                for method, metrics in method_totals.items()
            }

            report["experiments"][str(depth)] = {
                "aggregate": aggregate,
                "cases": case_results,
            }

            print("\nAggregate across eligible cases:")
            for method, values in aggregate.items():
                print(
                    f"  {method:14s} "
                    f"Recall@5={values['recall_at_5']:.3f} "
                    f"MRR={values['mrr']:.3f} "
                    f"nDCG@5={values['ndcg_at_5']:.3f}"
                )

        print("\n" + "=" * 72)
        print("COMPARISON: DEPTH 10 -> DEPTH 20")

        for method in (
            "bm25",
            "semantic",
            "hybrid_rrf",
            "cross_encoder",
        ):
            print(f"\n{method}")
            for metric in ("recall_at_5", "mrr", "ndcg_at_5"):
                before = report["experiments"]["10"]["aggregate"][method][metric]
                after = report["experiments"]["20"]["aggregate"][method][metric]
                print(
                    f"  {metric}: {before:.3f} -> {after:.3f} "
                    f"(delta {after - before:+.3f})"
                )

        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nReport saved: {REPORT_PATH}")
        print("Full top-20 diagnostic rankings saved.")
        print("Existing benchmark report was not modified.")

    finally:
        client.close()


if __name__ == "__main__":
    main()

# -*- coding: cp1252 -*-
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
BENCHMARK_PATH = DATA / "retrieval_benchmark_v3.json"
REPORT_PATH = DATA / "retrieval_benchmark_v3_results.json"
QDRANT_PATH = ROOT / "data" / "vector_store" / "qdrant_local"

COLLECTION = "nexagraph_siemens_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

TOP_K = 5
CANDIDATE_LIMIT = 10
RRF_K = 60
MAX_SEQUENCE_LENGTH = 256


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def key(record):
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
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def bm25_search(chunks, query, limit):
    documents = []
    tokenized_documents = []

    for chunk in chunks:
        terms = tokenize(chunk.get("text", ""))
        if terms:
            documents.append(chunk)
            tokenized_documents.append(terms)

    query_tokens = tokenize(query)
    if not documents or not query_tokens:
        return []

    model = BM25Okapi(tokenized_documents)
    scores = model.get_scores(query_tokens)
    ranked = []

    for index, score in enumerate(scores):
        if not set(query_tokens).intersection(tokenized_documents[index]):
            continue

        item = dict(documents[index])
        item["_score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda item: item["_score"], reverse=True)
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
        dict(point.payload or {})
        for point in result.points
        if point.payload
    ]


def rrf(rankings):
    fused = {}

    for ranking in rankings:
        for rank, record in enumerate(ranking, start=1):
            chunk_id = key(record)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "record": record,
                    "score": 0.0,
                }

            fused[chunk_id]["score"] += 1.0 / (RRF_K + rank)

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
    scored = []
    input_names = {item.name for item in session.get_inputs()}

    for record in candidates:
        encoded = tokenizer.encode(query, record.get("text", ""))

        inputs = {
            "input_ids": np.asarray(
                [encoded.ids],
                dtype=np.int64,
            ),
            "attention_mask": np.asarray(
                [encoded.attention_mask],
                dtype=np.int64,
            ),
        }

        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.asarray(
                [encoded.type_ids],
                dtype=np.int64,
            )

        output = session.run(None, inputs)[0]
        score = float(np.asarray(output).reshape(-1)[0])
        scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [record for _, record in scored]


def evidence_key(item):
    return key(item)


def metrics(ranked, expected, k=TOP_K):
    expected_keys = {evidence_key(item) for item in expected}
    if not expected_keys:
        raise ValueError("Cannot calculate metrics without expected evidence")

    retrieved_keys = [
        evidence_key(item)
        for item in ranked[:k]
    ]

    hits = [
        1 if item in expected_keys else 0
        for item in retrieved_keys
    ]

    found = set(retrieved_keys) & expected_keys
    recall = len(found) / len(expected_keys)

    reciprocal_rank = next(
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

    ideal_count = min(len(expected_keys), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    missing = [
        {
            "source_file": item.get("source_file"),
            "page_number": item.get("page_number"),
            "chunk_index": item.get("chunk_index"),
        }
        for item in expected
        if evidence_key(item) not in found
    ]

    return {
        "recall_at_5": recall,
        "mrr": reciprocal_rank,
        "ndcg_at_5": ndcg,
        "expected_evidence_count": len(expected_keys),
        "retrieved_evidence_count": len(found),
        "missing_evidence": missing,
        "top_5": [
            {
                "source_file": item.get("source_file"),
                "page_number": item.get("page_number"),
                "chunk_index": item.get("chunk_index"),
            }
            for item in ranked[:k]
        ],
    }


def is_eligible_for_aggregate(case):
    status = str(case.get("status", "")).lower()

    return (
        "needs_more" not in status
        and case.get("evaluation_eligible", True) is True
    )


def validate_benchmark(corpus, benchmark):
    corpus_keys = {key(item) for item in corpus}
    errors = []

    for case in benchmark.get("cases", []):
        case_id = case.get("case_id", "<unknown>")
        project_id = case.get("project_id")
        expected = case.get("expected_evidence", [])

        if not expected:
            errors.append(f"{case_id}: no expected evidence")
            continue

        for evidence in expected:
            if key(evidence) not in corpus_keys:
                errors.append(
                    f"{case_id}: evidence is not in corpus: "
                    f"page={evidence.get('page_number')} "
                    f"chunk={evidence.get('chunk_index')}"
                )

            if evidence.get("project_id") != project_id:
                errors.append(
                    f"{case_id}: cross-project evidence detected"
                )

    if errors:
        raise ValueError(
            "Benchmark validation failed:\n- " + "\n- ".join(errors)
        )


def main():
    for path in (CORPUS_PATH, BENCHMARK_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    corpus = load_jsonl(CORPUS_PATH)
    benchmark = json.loads(
        BENCHMARK_PATH.read_text(encoding="utf-8")
    )

    validate_benchmark(corpus, benchmark)

    eligible_cases = [
        case for case in benchmark["cases"]
        if is_eligible_for_aggregate(case)
    ]
    excluded_cases = [
        case for case in benchmark["cases"]
        if not is_eligible_for_aggregate(case)
    ]

    if not eligible_cases:
        raise ValueError(
            "No benchmark cases are eligible for aggregate metrics. "
            "Review evidence labels before evaluation."
        )

    print(f"Corpus records: {len(corpus)}")
    print(f"Total benchmark cases: {len(benchmark['cases'])}")
    print(f"Eligible cases: {len(eligible_cases)}")
    print(f"Excluded from aggregate: {len(excluded_cases)}")
    for case in excluded_cases:
        print(
            f"  EXCLUDED: {case['case_id']} "
            f"(status={case.get('status', 'unspecified')})"
        )

    print("Loading embedding model and local Qdrant...")
    embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    client = QdrantClient(path=str(QDRANT_PATH))

    report = {
        "benchmark_version": benchmark.get("benchmark_version"),
        "top_k": TOP_K,
        "total_case_count": len(benchmark["cases"]),
        "eligible_case_count": len(eligible_cases),
        "excluded_case_ids": [
            case["case_id"] for case in excluded_cases
        ],
        "cases": [],
        "aggregate": {},
        "limitations": [
            "Metrics measure retrieval of labeled chunks, not answer factuality.",
            "Expected evidence labels may not cover every valid passage.",
            "Excluded cases remain visible but do not affect aggregate metrics.",
            "The cross-encoder reranks only the hybrid candidate pool.",
            "Exact chunk matching can penalize adjacent chunks containing "
            "overlapping or complementary evidence.",
        ],
    }

    try:
        tokenizer, session = load_reranker()
        methods = (
            "bm25",
            "semantic",
            "hybrid_rrf",
            "cross_encoder",
        )
        totals = {
            method: {
                "recall_at_5": [],
                "mrr": [],
                "ndcg_at_5": [],
            }
            for method in methods
        }

        for case in benchmark["cases"]:
            project_id = case["project_id"]
            query = case["question"]

            project_chunks = [
                item for item in corpus
                if item.get("project_id") == project_id
                and item.get("text", "").strip()
            ]

            bm25 = bm25_search(
                project_chunks,
                query,
                CANDIDATE_LIMIT,
            )
            semantic = semantic_search(
                client,
                embedder,
                project_id,
                query,
                CANDIDATE_LIMIT,
            )
            hybrid = rrf([bm25, semantic])
            reranked = rerank(
                tokenizer,
                session,
                query,
                hybrid[:CANDIDATE_LIMIT],
            )

            rankings = {
                "bm25": bm25,
                "semantic": semantic,
                "hybrid_rrf": hybrid,
                "cross_encoder": reranked,
            }

            eligible = is_eligible_for_aggregate(case)
            case_report = {
                "case_id": case["case_id"],
                "project_id": project_id,
                "question": query,
                "benchmark_status": case.get("status"),
                "included_in_aggregate": eligible,
                "methods": {},
            }

            print("\n" + "=" * 76)
            print(f"{case['case_id']}: {query}")
            print(
                "Aggregate eligibility: "
                + ("YES" if eligible else "NO — diagnostic only")
            )

            for method, ranked in rankings.items():
                result = metrics(
                    ranked,
                    case["expected_evidence"],
                )
                case_report["methods"][method] = result

                if eligible:
                    for metric_name in totals[method]:
                        totals[method][metric_name].append(
                            result[metric_name]
                        )

                print(
                    f"{method:14s} "
                    f"Recall@5={result['recall_at_5']:.3f} "
                    f"MRR={result['mrr']:.3f} "
                    f"nDCG@5={result['ndcg_at_5']:.3f} "
                    f"found={result['retrieved_evidence_count']}/"
                    f"{result['expected_evidence_count']}"
                )

                if result["missing_evidence"]:
                    missing = ", ".join(
                        f"p{item['page_number']}/c{item['chunk_index']}"
                        for item in result["missing_evidence"]
                    )
                    print(f"  Missing expected evidence: {missing}")

            report["cases"].append(case_report)

        for method, values in totals.items():
            report["aggregate"][method] = {
                metric: (
                    sum(scores) / len(scores)
                    if scores else 0.0
                )
                for metric, scores in values.items()
            }

        print("\n" + "=" * 76)
        print(
            "AGGREGATE METRICS "
            f"(macro-average across {len(eligible_cases)} eligible cases)"
        )

        for method, values in report["aggregate"].items():
            print(
                f"{method:14s} "
                f"Recall@5={values['recall_at_5']:.3f} "
                f"MRR={values['mrr']:.3f} "
                f"nDCG@5={values['ndcg_at_5']:.3f}"
            )

        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\nDetailed report: {REPORT_PATH}")
        print("PASS: benchmark references validated.")
        print("NOTE: retrieval metrics do not prove answer or citation correctness.")

    finally:
        client.close()


if __name__ == "__main__":
    main()

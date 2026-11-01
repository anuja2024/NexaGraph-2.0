import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient
from tokenizers import Tokenizer
import onnxruntime as ort

from evaluate_hybrid import (
    CASES,
    CANDIDATE_LIMIT,
    COLLECTION_NAME,
    MODEL_NAME,
    QDRANT_PATH,
    RESULT_LIMIT,
    bm25_search,
    load_chunks,
    reciprocal_rank_fusion,
    semantic_search,
    validate_metadata,
)
from fastembed import TextEmbedding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RERANKER_REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_CANDIDATE_LIMIT = 10
MAX_SEQUENCE_LENGTH = 256


def load_reranker():
    print(f"Loading ONNX reranker: {RERANKER_REPO}")

    model_path = hf_hub_download(
        repo_id=RERANKER_REPO,
        filename="onnx/model.onnx",
    )
    tokenizer_path = hf_hub_download(
        repo_id=RERANKER_REPO,
        filename="tokenizer.json",
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    return tokenizer, session


def score_passage(tokenizer, session, query, passage):
    encoded = tokenizer.encode(query, passage)

    inputs = {
        "input_ids": np.asarray(
            [encoded.ids], dtype=np.int64
        ),
        "attention_mask": np.asarray(
            [encoded.attention_mask], dtype=np.int64
        ),
    }

    input_names = {
        item.name for item in session.get_inputs()
    }

    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = np.asarray(
            [encoded.type_ids], dtype=np.int64
        )

    output = session.run(None, inputs)[0]
    return float(np.asarray(output).reshape(-1)[0])


def main():
    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant storage not found: {QDRANT_PATH}"
        )

    print("Loading corpus...")
    all_chunks = load_chunks()
    print("Corpus records:", len(all_chunks))

    print("Loading embedding model:", MODEL_NAME)
    embedding_model = TextEmbedding(model_name=MODEL_NAME)

    tokenizer, reranker_session = load_reranker()
    client = QdrantClient(path=str(QDRANT_PATH))

    passed = 0
    failed = 0

    def check(condition, message):
        nonlocal passed, failed
        if condition:
            passed += 1
            print("PASS:", message)
        else:
            failed += 1
            print("FAIL:", message)

    try:
        check(
            client.collection_exists(COLLECTION_NAME),
            "Qdrant collection exists",
        )

        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Missing collection: {COLLECTION_NAME}"
            )

        indexed_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        check(
            indexed_count == len(all_chunks),
            "Qdrant count matches corpus count",
        )

        for case in CASES:
            project_id = case["project_id"]
            query = case["query"]

            print("\n" + "=" * 78)
            print("PROJECT:", project_id)
            print("QUERY:", query)

            project_chunks = [
                item for item in all_chunks
                if item.get("project_id") == project_id
                and isinstance(item.get("text"), str)
                and item["text"].strip()
            ]

            bm25_results = bm25_search(
                project_chunks,
                query,
                CANDIDATE_LIMIT,
            )
            semantic_results = semantic_search(
                client,
                embedding_model,
                project_id,
                query,
                CANDIDATE_LIMIT,
            )

            check(bool(bm25_results), f"{project_id}: BM25 candidates")
            check(
                bool(semantic_results),
                f"{project_id}: semantic candidates",
            )

            fused = reciprocal_rank_fusion(
                [bm25_results, semantic_results]
            )

            # Keep only the best unique RRF candidates.
            candidates = []
            seen = set()

            for item in fused:
                record = item["record"]
                key = (
                    record.get("project_id"),
                    record.get("document_id"),
                    record.get("source_file"),
                    record.get("page_number"),
                    record.get("chunk_index"),
                    record.get("text", ""),
                )

                if key in seen:
                    continue

                seen.add(key)
                candidates.append(item)

                if len(candidates) >= RERANK_CANDIDATE_LIMIT:
                    break

            check(
                bool(candidates),
                f"{project_id}: unique RRF candidates available",
            )

            for item in candidates:
                record = item["record"]
                item["reranker_score"] = score_passage(
                    tokenizer,
                    reranker_session,
                    query,
                    record["text"],
                )

            reranked = sorted(
                candidates,
                key=lambda item: item["reranker_score"],
                reverse=True,
            )[:RESULT_LIMIT]

            print("\nRRF ranking before reranking:")

            for rank, item in enumerate(
                candidates[:RESULT_LIMIT], start=1
            ):
                record = item["record"]
                print(
                    f"{rank}. RRF={item['rrf_score']:.6f} | "
                    f"{record.get('source_file')} | "
                    f"page {record.get('page_number')}"
                )

            print("\nCross-encoder ranking:")

            for rank, item in enumerate(reranked, start=1):
                record = item["record"]
                text = " ".join(record["text"].split())

                print(f"\n{rank}. Reranker score: {item['reranker_score']:.6f}")
                print("   File:", record.get("source_file"))
                print("   Page:", record.get("page_number"))
                print("   Document ID:", record.get("document_id"))
                print("   Passage:", text[:300])

                check(
                    record.get("project_id") == project_id,
                    f"{project_id}: reranked result stays in project",
                )
                check(
                    validate_metadata(record),
                    f"{project_id}: reranked result has citation metadata",
                )

            check(
                len(reranked) <= RESULT_LIMIT,
                f"{project_id}: reranked result count respects limit",
            )

        print("\n" + "=" * 78)
        print("Checks passed:", passed)
        print("Checks failed:", failed)

        if failed:
            raise SystemExit("RERANKER EVALUATION FAILED")

        print("RERANKER EVALUATION PASSED")
        print(
            "Important: structural checks do not prove factual "
            "relevance or citation correctness."
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()

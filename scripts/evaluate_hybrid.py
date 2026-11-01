import json
import re
from pathlib import Path

from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "staging" / "siemens_pdf" / "chunks.jsonl"
QDRANT_PATH = PROJECT_ROOT / "data" / "vector_store" / "qdrant_local"

COLLECTION_NAME = "nexagraph_siemens_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
RRF_K = 60
CANDIDATE_LIMIT = 10
RESULT_LIMIT = 5

CASES = [
    {
        "project_id": "corporate",
        "query": "How has Siemens performed financially?",
    },
    {
        "project_id": "sustainability",
        "query": "How does Siemens reduce carbon emissions?",
    },
    {
        "project_id": "industrial_automation",
        "query": (
            "How can industrial automation systems be protected "
            "against cyber threats?"
        ),
    },
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Corpus not found: {CHUNKS_PATH}")

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on corpus line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid record on corpus line {line_number}"
                )

            chunks.append(record)

    return chunks


def chunk_key(record: dict) -> tuple:
    """Identify the same source chunk across retrieval methods."""
    return (
        record.get("project_id"),
        record.get("document_id"),
        record.get("source_file"),
        record.get("page_number"),
        record.get("chunk_index"),
        record.get("text", ""),
    )


def validate_metadata(record: dict) -> bool:
    return (
        isinstance(record.get("project_id"), str)
        and bool(record.get("project_id"))
        and isinstance(record.get("document_id"), str)
        and bool(record.get("document_id"))
        and isinstance(record.get("source_file"), str)
        and bool(record.get("source_file"))
        and isinstance(record.get("page_number"), int)
        and record.get("page_number") >= 1
        and isinstance(record.get("text"), str)
        and bool(record.get("text").strip())
    )


def bm25_search(
    project_chunks: list[dict],
    query: str,
    limit: int,
) -> list[dict]:
    query_tokens = tokenize(query)

    indexed_chunks = []
    tokenized_documents = []

    for chunk in project_chunks:
        tokens = tokenize(chunk["text"])

        if tokens:
            indexed_chunks.append(chunk)
            tokenized_documents.append(tokens)

    if not query_tokens or not indexed_chunks:
        return []

    model = BM25Okapi(tokenized_documents)
    scores = model.get_scores(query_tokens)
    query_terms = set(query_tokens)
    ranked = []

    for index, score in enumerate(scores):
        matched_terms = query_terms.intersection(
            tokenized_documents[index]
        )

        if not matched_terms:
            continue

        record = dict(indexed_chunks[index])
        record["_bm25_score"] = float(score)
        record["_matched_terms"] = sorted(matched_terms)
        ranked.append(record)

    ranked.sort(
        key=lambda item: item["_bm25_score"],
        reverse=True,
    )

    return ranked[:limit]


def semantic_search(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    project_id: str,
    query: str,
    limit: int,
) -> list[dict]:
    query_vectors = list(embedding_model.query_embed(query))

    if len(query_vectors) != 1:
        raise RuntimeError("Expected exactly one query embedding")

    query_vector = [float(value) for value in query_vectors[0]]

    project_filter = Filter(
        must=[
            FieldCondition(
                key="project_id",
                match=MatchValue(value=project_id),
            )
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=project_filter,
        limit=limit,
        with_payload=True,
    )

    ranked = []

    for point in response.points:
        payload = point.payload or {}
        record = dict(payload)
        record["_semantic_score"] = float(point.score)
        ranked.append(record)

    return ranked


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    rrf_k: int = RRF_K,
) -> list[dict]:
    fused = {}

    for ranking_name, ranking in zip(
        ("BM25", "Semantic"),
        rankings,
    ):
        for rank, record in enumerate(ranking, start=1):
            key = chunk_key(record)

            if key not in fused:
                fused[key] = {
                    "record": record,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None,
                }

            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)
            fused[key][f"{ranking_name.lower()}_rank"] = rank

            # Prefer the corpus record for canonical source metadata.
            if ranking_name == "BM25":
                fused[key]["record"] = record

    results = list(fused.values())
    results.sort(key=lambda item: item["rrf_score"], reverse=True)

    return results


def main() -> None:
    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant storage not found: {QDRANT_PATH}"
        )

    print("Loading corpus...")
    all_chunks = load_chunks()
    print(f"Corpus records: {len(all_chunks)}")

    print(f"Loading embedding model: {MODEL_NAME}")
    embedding_model = TextEmbedding(model_name=MODEL_NAME)
    client = QdrantClient(path=str(QDRANT_PATH))

    passed = 0
    failed = 0

    def check(condition: bool, message: str) -> None:
        nonlocal passed, failed

        if condition:
            passed += 1
            print(f"PASS: {message}")
        else:
            failed += 1
            print(f"FAIL: {message}")

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Qdrant collection not found: {COLLECTION_NAME}"
            )

        indexed_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print(f"Qdrant points: {indexed_count}")
        check(
            indexed_count == len(all_chunks),
            "Qdrant point count matches corpus record count",
        )

        for case in CASES:
            project_id = case["project_id"]
            query = case["query"]

            project_chunks = [
                chunk
                for chunk in all_chunks
                if chunk.get("project_id") == project_id
                and isinstance(chunk.get("text"), str)
                and chunk["text"].strip()
            ]

            print("\n" + "=" * 78)
            print(f"PROJECT: {project_id}")
            print(f"QUERY: {query}")
            print(f"Project corpus records: {len(project_chunks)}")

            check(
                bool(project_chunks),
                f"{project_id}: project corpus is not empty",
            )

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

            check(
                bool(bm25_results),
                f"{project_id}: BM25 returned candidates",
            )
            check(
                bool(semantic_results),
                f"{project_id}: semantic search returned candidates",
            )

            for method, results in (
                ("BM25", bm25_results),
                ("Semantic", semantic_results),
            ):
                wrong_project = [
                    result
                    for result in results
                    if result.get("project_id") != project_id
                ]

                check(
                    not wrong_project,
                    f"{project_id}: {method} results stay in project",
                )

                check(
                    all(validate_metadata(result) for result in results),
                    f"{project_id}: {method} results have citation metadata",
                )

            fused_results = reciprocal_rank_fusion(
                [bm25_results, semantic_results]
            )[:RESULT_LIMIT]

            check(
                bool(fused_results),
                f"{project_id}: RRF returned fused candidates",
            )

            print("\nTop fused results:")

            for rank, item in enumerate(fused_results, start=1):
                record = item["record"]

                print(f"\n  Rank: {rank}")
                print(f"  RRF score: {item['rrf_score']:.6f}")
                print(f"  BM25 rank: {item['bm25_rank']}")
                print(f"  Semantic rank: {item['semantic_rank']}")
                print(f"  File: {record.get('source_file')}")
                print(f"  Page: {record.get('page_number')}")
                print(f"  Document ID: {record.get('document_id')}")

                text = " ".join(record.get("text", "").split())
                print(f"  Passage: {text[:350]}")

                check(
                    record.get("project_id") == project_id,
                    f"{project_id}: fused result belongs to requested project",
                )
                check(
                    validate_metadata(record),
                    f"{project_id}: fused result has source metadata",
                )

        print("\n" + "=" * 78)
        print(f"Checks passed: {passed}")
        print(f"Checks failed: {failed}")

        if failed:
            print("HYBRID RETRIEVAL TEST FAILED")
            raise SystemExit(1)

        print("HYBRID RETRIEVAL TEST PASSED")
        print(
            "Note: Passing checks validates pipeline behavior, "
            "not factual relevance or citation correctness."
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()

from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QDRANT_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "qdrant_local"
)

COLLECTION_NAME = "nexagraph_siemens_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
RESULT_LIMIT = 3

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
        "query": "How can industrial automation systems be protected against cyber threats?",
    },
]


def main() -> None:
    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            f"Qdrant storage not found: {QDRANT_PATH}"
        )

    print(f"Loading embedding model: {MODEL_NAME}")
    embedding_model = TextEmbedding(model_name=MODEL_NAME)

    client = QdrantClient(path=str(QDRANT_PATH))

    passed = 0
    failed = 0

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection not found: {COLLECTION_NAME}"
            )

        collection_info = client.get_collection(COLLECTION_NAME)
        actual_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        print(f"Collection points: {actual_count}")

        if actual_count != 2251:
            print(
                "WARNING: The current corpus count differs "
                "from the expected 2251 chunks."
            )

        for case in CASES:
            project_id = case["project_id"]
            query = case["query"]

            print("\n" + "=" * 70)
            print(f"PROJECT: {project_id}")
            print(f"QUERY: {query}")

            query_vectors = list(
                embedding_model.query_embed(query)
            )

            if len(query_vectors) != 1:
                raise RuntimeError(
                    "Expected exactly one query embedding"
                )

            query_vector = [
                float(value) for value in query_vectors[0]
            ]

            if len(query_vector) != VECTOR_SIZE:
                raise RuntimeError(
                    f"Expected {VECTOR_SIZE} dimensions, "
                    f"got {len(query_vector)}"
                )

            # Apply the project filter inside Qdrant retrieval.
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
                limit=RESULT_LIMIT,
                with_payload=True,
            )

            results = response.points

            print(f"Results: {len(results)}")

            if results:
                passed += 1
                print("PASS: Qdrant returned search results")
            else:
                failed += 1
                print("FAIL: Qdrant returned no results")

            for rank, point in enumerate(results, start=1):
                payload = point.payload or {}

                print(f"\nResult {rank}")
                print(f"  Similarity score: {point.score}")
                print(f"  Project: {payload.get('project_id')}")
                print(f"  File: {payload.get('source_file')}")
                print(f"  Page: {payload.get('page_number')}")
                print(f"  Document ID: {payload.get('document_id')}")

                text = payload.get("text", "")
                print(
                    "  Passage: "
                    + " ".join(str(text).split())[:400]
                )

                if payload.get("project_id") == project_id:
                    passed += 1
                    print("  PASS: Project filter respected")
                else:
                    failed += 1
                    print("  FAIL: Project filter violation")

                metadata_valid = (
                    isinstance(payload.get("source_file"), str)
                    and bool(payload.get("source_file"))
                    and isinstance(payload.get("page_number"), int)
                    and payload.get("page_number") >= 1
                    and isinstance(payload.get("document_id"), str)
                    and bool(payload.get("document_id"))
                    and isinstance(payload.get("text"), str)
                    and bool(payload.get("text").strip())
                )

                if metadata_valid:
                    passed += 1
                    print("  PASS: Source metadata present")
                else:
                    failed += 1
                    print("  FAIL: Source metadata incomplete")

        print("\n" + "=" * 70)
        print(f"Checks passed: {passed}")
        print(f"Checks failed: {failed}")

        if failed:
            print("SEMANTIC SEARCH TEST FAILED")
            raise SystemExit(1)

        print("SEMANTIC SEARCH TEST PASSED")

    finally:
        client.close()


if __name__ == "__main__":
    main()

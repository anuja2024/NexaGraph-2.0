
import json
import uuid
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "siemens_pdf"
    / "chunks.jsonl"
)

QDRANT_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "qdrant_local"
)

COLLECTION_NAME = "nexagraph_siemens_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
BATCH_SIZE = 64

EXPECTED_PROJECTS = {
    "corporate",
    "sustainability",
    "industrial_automation",
}


def load_chunks() -> list[dict]:
    """Load and validate the active chunk corpus."""
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            f"Chunk corpus not found: {CHUNKS_PATH}"
        )

    chunks = []

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from error

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Chunk on line {line_number} is not an object"
                )

            required_fields = [
                "document_id",
                "project_id",
                "source_file",
                "page_number",
                "chunk_index",
                "text",
            ]

            missing = [
                field
                for field in required_fields
                if field not in chunk
            ]

            if missing:
                raise ValueError(
                    f"Chunk on line {line_number} is missing "
                    f"fields: {missing}"
                )

            if chunk["project_id"] not in EXPECTED_PROJECTS:
                raise ValueError(
                    f"Unexpected project ID on line {line_number}: "
                    f"{chunk['project_id']}"
                )

            if (
                not isinstance(chunk["text"], str)
                or not chunk["text"].strip()
            ):
                raise ValueError(
                    f"Empty or invalid text on line {line_number}"
                )

            if (
                not isinstance(chunk["page_number"], int)
                or chunk["page_number"] < 1
            ):
                raise ValueError(
                    f"Invalid page number on line {line_number}"
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError("The chunk corpus is empty")

    return chunks


def stable_point_id(chunk: dict) -> str:
    """Generate a deterministic UUID for a source chunk."""
    identity = (
        f"{chunk['document_id']}|"
        f"{chunk['project_id']}|"
        f"{chunk['source_file']}|"
        f"{chunk['page_number']}|"
        f"{chunk['chunk_index']}"
    )

    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def main() -> None:
    chunks = load_chunks()

    print(f"Corpus loaded: {len(chunks)} chunks")

    project_counts = {}
    for chunk in chunks:
        project_id = chunk["project_id"]
        project_counts[project_id] = (
            project_counts.get(project_id, 0) + 1
        )

    print("Project distribution:")
    for project_id, count in sorted(project_counts.items()):
        print(f"  {project_id}: {count}")

    QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(QDRANT_PATH))

    try:
        # Never overwrite an existing collection automatically.
        if client.collection_exists(COLLECTION_NAME):
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' already exists. "
                "No indexing was performed. Inspect the collection "
                "before deciding whether to reuse or rebuild it."
            )

        print(f"\nLoading embedding model: {MODEL_NAME}")
        embedding_model = TextEmbedding(model_name=MODEL_NAME)

        print("Creating local Qdrant collection...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        indexed = 0

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            texts = [chunk["text"] for chunk in batch]

            embeddings = list(embedding_model.embed(texts))

            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch in batch "
                    f"starting at chunk {start}"
                )

            points = []

            for chunk, embedding in zip(batch, embeddings):
                vector = [float(value) for value in embedding]

                if len(vector) != VECTOR_SIZE:
                    raise RuntimeError(
                        f"Expected {VECTOR_SIZE} vector dimensions; "
                        f"received {len(vector)}"
                    )

                payload = {
                    "document_id": chunk["document_id"],
                    "project_id": chunk["project_id"],
                    "source_file": chunk["source_file"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                }

                points.append(
                    PointStruct(
                        id=stable_point_id(chunk),
                        vector=vector,
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

            indexed += len(points)
            print(f"Indexed {indexed}/{len(chunks)} chunks")

        final_count = client.count(
            collection_name=COLLECTION_NAME,
            exact=True,
        ).count

        if final_count != len(chunks):
            raise RuntimeError(
                f"Index verification failed: expected {len(chunks)} "
                f"points, found {final_count}"
            )

        print("\nQDRANT INDEXING PASSED")
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Vector dimensions: {VECTOR_SIZE}")
        print("Distance metric: cosine")
        print(f"Indexed points: {final_count}")
        print(f"Persistent storage: {QDRANT_PATH}")

    finally:
        client.close()


if __name__ == "__main__":
    main()

import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "siemens_pdf"
    / "chunks.jsonl"
)

EXPECTED_PROJECTS = {
    "corporate",
    "sustainability",
    "industrial_automation",
}

BATCH_SIZE = 200


def stable_chunk_id(chunk: dict) -> str:
    """Match the deterministic UUID used by the Qdrant index."""
    identity = (
        f"{chunk['document_id']}|"
        f"{chunk['project_id']}|"
        f"{chunk['source_file']}|"
        f"{chunk['page_number']}|"
        f"{chunk['chunk_index']}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Corpus not found: {CHUNKS_PATH}")

    chunks = []
    seen_ids = set()

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            required = {
                "document_id",
                "project_id",
                "source_file",
                "page_number",
                "chunk_index",
                "text",
            }

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Record on line {line_number} is not an object"
                )

            missing = required - chunk.keys()
            if missing:
                raise ValueError(
                    f"Missing fields on line {line_number}: "
                    f"{sorted(missing)}"
                )

            if chunk["project_id"] not in EXPECTED_PROJECTS:
                raise ValueError(
                    f"Unexpected project on line {line_number}"
                )

            if not all(
                isinstance(chunk[field], str) and chunk[field].strip()
                for field in ("document_id", "source_file", "text")
            ):
                raise ValueError(
                    f"Invalid text or identifier on line {line_number}"
                )

            if (
                type(chunk["page_number"]) is not int
                or chunk["page_number"] < 1
            ):
                raise ValueError(
                    f"Invalid page number on line {line_number}"
                )

            if (
                type(chunk["chunk_index"]) is not int
                or chunk["chunk_index"] < 0
            ):
                raise ValueError(
                    f"Invalid chunk index on line {line_number}"
                )

            record = {
                "chunk_id": stable_chunk_id(chunk),
                "document_id": chunk["document_id"],
                "project_id": chunk["project_id"],
                "source_file": chunk["source_file"],
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }

            if record["chunk_id"] in seen_ids:
                raise ValueError(
                    f"Duplicate chunk identity on line {line_number}"
                )

            seen_ids.add(record["chunk_id"])
            chunks.append(record)

    if not chunks:
        raise ValueError("The corpus contains no chunks")

    return chunks


def create_constraints(session) -> None:
    statements = [
        """
        CREATE CONSTRAINT nexagraph_project_id IF NOT EXISTS
        FOR (n:Project) REQUIRE n.project_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT nexagraph_document_id IF NOT EXISTS
        FOR (n:Document) REQUIRE n.document_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT nexagraph_chunk_id IF NOT EXISTS
        FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE
        """,
    ]

    for statement in statements:
        session.run(statement).consume()


def ingest_batch(tx, records: list[dict]) -> int:
    result = tx.run(
        """
        UNWIND $rows AS row

        MERGE (p:Project {project_id: row.project_id})
        SET p.name = row.project_id

        MERGE (d:Document {document_id: row.document_id})
        SET d.project_id = row.project_id,
            d.source_file = row.source_file

        MERGE (d)-[:IN_PROJECT]->(p)

        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c.project_id = row.project_id,
            c.document_id = row.document_id,
            c.source_file = row.source_file,
            c.page_number = row.page_number,
            c.chunk_index = row.chunk_index,
            c.text = row.text

        MERGE (d)-[:HAS_CHUNK]->(c)

        RETURN count(c) AS processed
        """,
        rows=records,
    )

    record = result.single()
    return record["processed"]


def main() -> None:
    load_dotenv(
        dotenv_path=PROJECT_ROOT / ".env",
        override=True,
    )

    uri = __import__("os").environ.get("NEO4J_URI")
    user = __import__("os").environ.get("NEO4J_USER")
    password = __import__("os").environ.get("NEO4J_PASSWORD")
    database = __import__("os").environ.get(
        "NEO4J_DATABASE", "neo4j"
    )

    if not all((uri, user, password)):
        raise RuntimeError(
            "NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD "
            "must be configured in .env"
        )

    chunks = load_chunks()
    print(f"Validated corpus: {len(chunks)} chunks")

    project_counts = {}
    for chunk in chunks:
        project_id = chunk["project_id"]
        project_counts[project_id] = (
            project_counts.get(project_id, 0) + 1
        )

    print("Project distribution:")
    for project_id, count in sorted(project_counts.items()):
        print(f"  {project_id}: {count}")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        driver.verify_connectivity()

        with driver.session(database=database) as session:
            create_constraints(session)

            processed = 0
            for start in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[start:start + BATCH_SIZE]
                count = session.execute_write(ingest_batch, batch)
                processed += count
                print(f"Ingested {processed}/{len(chunks)} chunks")

            verification = session.run(
                """
                MATCH (p:Project)
                WITH count(p) AS projects
                MATCH (d:Document)
                WITH projects, count(d) AS documents
                MATCH (c:Chunk)
                RETURN projects, documents, count(c) AS chunks
                """
            ).single()

            print("\nNEO4J INGESTION COMPLETED")
            print(f"Project nodes: {verification['projects']}")
            print(f"Document nodes: {verification['documents']}")
            print(f"Chunk nodes: {verification['chunks']}")

            if verification["chunks"] < len(chunks):
                raise RuntimeError(
                    "Verification failed: fewer graph chunks than "
                    "the active corpus. Existing unrelated graph data "
                    "was not deleted."
                )

            print("Chunk-count verification: PASSED")

    finally:
        driver.close()


if __name__ == "__main__":
    main()

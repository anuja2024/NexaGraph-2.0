import json
from pathlib import Path

path = Path("data/staging/siemens_pdf/chunks.jsonl")
targets = {
    ("corporate", 22, 0),
    ("corporate", 22, 2),
    ("corporate", 22, 3),
    ("corporate", 22, 5),
    ("corporate", 22, 7),
    ("corporate", 5, 0),
    ("corporate", 5, 1),
}

with path.open("r", encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file if line.strip()]

for chunk in chunks:
    key = (
        chunk.get("project_id"),
        chunk.get("page_number"),
        chunk.get("chunk_index"),
    )
    if key in targets:
        print("\n" + "=" * 72)
        print(
            f"Project: {key[0]} | Page: {key[1]} | "
            f"Chunk: {key[2]} | File: {chunk.get('source_file')}"
        )
        print(chunk.get("text", ""))

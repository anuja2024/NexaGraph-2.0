import json
from pathlib import Path

path = Path("data/staging/siemens_pdf/chunks.jsonl")
targets = {
    ("industrial_automation", 9, 0),
    ("industrial_automation", 17, 0),
    ("industrial_automation", 17, 1),
    ("industrial_automation", 34, 1),
    ("industrial_automation", 13, 0),
    ("industrial_automation", 13, 1),
}

with path.open("r", encoding="utf-8") as file:
    for line in file:
        if not line.strip():
            continue

        chunk = json.loads(line)
        key = (
            chunk.get("project_id"),
            chunk.get("page_number"),
            chunk.get("chunk_index"),
        )

        if key in targets:
            print("\n" + "=" * 72)
            print(
                f"Page {key[1]} | Chunk {key[2]} | "
                f"{chunk.get('source_file')}"
            )
            print(chunk.get("text", ""))

import json
from pathlib import Path

path = Path("data/staging/siemens_pdf/retrieval_benchmark.json")
benchmark = json.loads(path.read_text(encoding="utf-8"))

for case in benchmark["cases"]:
    if case["case_id"].startswith("industrial_"):
        print("\n" + "=" * 72)
        print("Case:", case["case_id"])
        print("Question:", case["question"])
        print("Status:", case.get("status"))

        for item in case["expected_evidence"]:
            print(
                f"  page={item['page_number']} "
                f"chunk={item['chunk_index']} "
                f"file={item['source_file']}"
            )

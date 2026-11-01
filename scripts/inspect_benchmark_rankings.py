import json
from pathlib import Path

path = Path("data/staging/siemens_pdf/retrieval_benchmark_results.json")
report = json.loads(path.read_text(encoding="utf-8"))

for case in report["cases"]:
    print()
    print("=" * 72)
    print(case["case_id"])

    for method, result in case["methods"].items():
        print()
        print(method + ":")

        for rank, item in enumerate(result["top_5"], start=1):
            print(
                f"  {rank}. page={item['page_number']} "
                f"chunk={item['chunk_index']} "
                f"file={item['source_file']}"
            )

import json
from pathlib import Path

path = Path("data/staging/siemens_pdf/candidate_depth_experiment.json")
report = json.loads(path.read_text(encoding="utf-8"))

for depth in ("10", "20"):
    print("\n" + "=" * 60)
    print("Candidate depth:", depth)

    for case in report["experiments"][depth]["cases"]:
        if case["case_id"] != "industrial_network_security":
            continue

        for method in ("hybrid_rrf", "cross_encoder"):
            print("\n" + method)
            for rank, item in enumerate(
                case["methods"][method]["top_5"], start=1
            ):
                print(
                    f"  {rank}. page={item['page_number']} "
                    f"chunk={item['chunk_index']} "
                    f"file={item['source_file']}"
                )

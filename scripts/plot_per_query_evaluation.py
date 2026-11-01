
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "staging" / "siemens_pdf"
OUTPUT_DIR = ROOT / "assets" / "evaluation"

RESULTS_FILE = DATA_DIR / "retrieval_benchmark_results.json"

METHODS = {
    "bm25": "BM25",
    "semantic": "Semantic",
    "hybrid_rrf": "Hybrid RRF",
    "cross_encoder": "Cross-encoder",
}

METRICS = {
    "recall_at_5": {
        "title": "Per-Query Recall@5",
        "filename": "per_query_recall_at_5.png",
        "label": "Recall@5",
        "format": ".1%",
    },
    "mrr": {
        "title": "Per-Query Mean Reciprocal Rank",
        "filename": "per_query_mrr.png",
        "label": "MRR",
        "format": ".3f",
    },
    "ndcg_at_5": {
        "title": "Per-Query nDCG@5",
        "filename": "per_query_ndcg_at_5.png",
        "label": "nDCG@5",
        "format": ".3f",
    },
}


# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------

def load_results():
    if not RESULTS_FILE.is_file():
        raise FileNotFoundError(RESULTS_FILE)

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report.get("cases"), list):
        raise ValueError("Missing benchmark cases in results JSON.")

    return report


# --------------------------------------------------
# EXTRACT ELIGIBLE CASES
# --------------------------------------------------

def extract_cases(report):
    cases = [
        case
        for case in report["cases"]
        if case.get("included_in_aggregate") is True
    ]

    if not cases:
        raise ValueError("No eligible benchmark cases found.")

    case_labels = []
    metric_values = {metric: [] for metric in METRICS}

    for case in cases:
        case_id = case.get("case_id", "unknown_case")
        methods = case.get("methods", {})

        case_labels.append(case_id)

        for method in METHODS:
            if method not in methods:
                raise ValueError(
                    f"Missing method '{method}' in {case_id}"
                )

            for metric in METRICS:
                if metric not in methods[method]:
                    raise ValueError(
                        f"Missing {metric} for {method} in {case_id}"
                    )

                value = float(methods[method][metric])

                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(
                        f"Invalid metric value: {case_id}, "
                        f"{method}, {metric} = {value}"
                    )

                metric_values[metric].append(value)

    return case_labels, metric_values


# --------------------------------------------------
# PLOT HEATMAP
# --------------------------------------------------

def plot_heatmap(case_labels, metric_values, metric, report):
    config = METRICS[metric]

    matrix = np.array(metric_values[metric]).reshape(
        len(case_labels),
        len(METHODS),
    )

    fig, ax = plt.subplots(
        figsize=(12, max(5, len(case_labels) * 1.15))
    )

    image = ax.imshow(
        matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(METHODS.values())

    ax.set_yticks(range(len(case_labels)))
    ax.set_yticklabels(case_labels)

    ax.set_title(config["title"], fontsize=16, pad=18)
    ax.set_xlabel("Retrieval Method", fontsize=11)
    ax.set_ylabel("Benchmark Question", fontsize=11)

    # Annotate every cell with its actual metric value.
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]

            label = (
                f"{value:.1%}"
                if metric == "recall_at_5"
                else f"{value:.3f}"
            )

            text_color = "white" if value >= 0.55 else "black"

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="normal",
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(config["label"])

    eligible = report.get("eligible_case_count", len(case_labels))
    total = report.get("total_case_count", "Unknown")

    fig.text(
        0.5,
        0.025,
        f"Initial benchmark | Eligible cases: {eligible}/{total} "
        "| Exact chunk-identity matching",
        ha="center",
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.25,
        right=0.91,
        top=0.88,
        bottom=0.16,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / config["filename"]

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    report = load_results()

    case_labels, metric_values = extract_cases(report)

    print("\nNEXAGRAPH PER-QUERY EVALUATION")
    print("=" * 75)

    print(f"Eligible questions: {len(case_labels)}")
    print(f"Total benchmark questions: {len(report['cases'])}")

    print("\nGENERATING PER-QUERY HEATMAPS")
    print("-" * 75)

    for metric in METRICS:
        plot_heatmap(
            case_labels,
            metric_values,
            metric,
            report,
        )

    print("\nSUCCESS: All three per-query heatmaps generated.")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

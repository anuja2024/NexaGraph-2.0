
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "staging" / "siemens_pdf"
OUTPUT_DIR = ROOT / "assets" / "evaluation"

RESULTS_FILE = DATA_DIR / "retrieval_benchmark_results.json"

METHOD_LABELS = {
    "bm25": "BM25",
    "semantic": "Semantic",
    "hybrid_rrf": "Hybrid RRF",
    "cross_encoder": "Cross-encoder",
}

EXPECTED_METRICS = (
    "recall_at_5",
    "mrr",
    "ndcg_at_5",
)

DPI = 300


# --------------------------------------------------
# LOAD AND VALIDATE RESULTS
# --------------------------------------------------

def load_results():
    if not RESULTS_FILE.is_file():
        raise FileNotFoundError(
            f"Evaluation results not found: {RESULTS_FILE}"
        )

    with RESULTS_FILE.open("r", encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise ValueError("Evaluation report must be a JSON object.")

    return report


def extract_metrics(report):
    aggregate = report.get("aggregate")

    if not isinstance(aggregate, dict) or not aggregate:
        raise ValueError("No aggregate metrics found.")

    methods = []
    recall = []
    mrr = []
    ndcg = []

    for method, label in METHOD_LABELS.items():
        if method not in aggregate:
            raise ValueError(
                f"Missing retrieval method in results: {method}"
            )

        values = aggregate[method]

        for metric in EXPECTED_METRICS:
            if metric not in values:
                raise ValueError(
                    f"Missing metric '{metric}' for {method}"
                )

            value = float(values[metric])

            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"Invalid {metric} value for {method}: {value}"
                )

        methods.append(label)
        recall.append(float(values["recall_at_5"]))
        mrr.append(float(values["mrr"]))
        ndcg.append(float(values["ndcg_at_5"]))

    return methods, recall, mrr, ndcg


# --------------------------------------------------
# CHART FORMATTING
# --------------------------------------------------

def add_value_labels(ax, bars, values, percentage=False):
    for bar, value in zip(bars, values):
        label = (
            f"{value * 100:.1f}%"
            if percentage
            else f"{value:.3f}"
        )

        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2,
                bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def add_benchmark_note(fig, report):
    eligible = report.get("eligible_case_count", "Unknown")
    total = report.get("total_case_count", "Unknown")

    fig.text(
        0.5,
        0.035,
        f"Initial benchmark | Eligible cases: {eligible}/{total} "
        "| Exact chunk-identity matching",
        ha="center",
        fontsize=9,
    )


def save_plot(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# PLOT 1: RECALL@5
# --------------------------------------------------

def plot_recall(methods, recall, report):
    fig, ax = plt.subplots(figsize=(10, 6))

    values = np.array(recall) * 100

    bars = ax.bar(methods, values, width=0.6)

    ax.set_title(
        "Retrieval Recall@5 Comparison",
        fontsize=16,
        pad=15,
    )
    ax.set_xlabel("Retrieval Method", fontsize=11)
    ax.set_ylabel("Recall@5 (%)", fontsize=11)

    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    add_value_labels(ax, bars, recall, percentage=True)
    add_benchmark_note(fig, report)

    fig.subplots_adjust(bottom=0.19, top=0.87)

    save_plot(fig, "recall_at_5.png")


# --------------------------------------------------
# PLOT 2: MRR AND NDCG@5
# --------------------------------------------------

def plot_ranking_quality(methods, mrr, ndcg, report):
    x = np.arange(len(methods))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11, 6))

    mrr_bars = ax.bar(
        x - width / 2,
        mrr,
        width,
        label="MRR",
    )

    ndcg_bars = ax.bar(
        x + width / 2,
        ndcg,
        width,
        label="nDCG@5",
    )

    ax.set_title(
        "Retrieval Ranking Quality Comparison",
        fontsize=16,
        pad=15,
    )

    ax.set_xlabel("Retrieval Method", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.set_ylim(0, 0.9)

    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    add_value_labels(ax, mrr_bars, mrr)
    add_value_labels(ax, ndcg_bars, ndcg)

    add_benchmark_note(fig, report)

    fig.subplots_adjust(bottom=0.19, top=0.87)

    save_plot(fig, "ranking_quality.png")


# --------------------------------------------------
# PLOT 3: COMBINED RETRIEVAL METRICS
# --------------------------------------------------

def plot_all_metrics(methods, recall, mrr, ndcg, report):
    x = np.arange(len(methods))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 6))

    recall_bars = ax.bar(
        x - width,
        recall,
        width,
        label="Recall@5",
    )

    mrr_bars = ax.bar(
        x,
        mrr,
        width,
        label="MRR",
    )

    ndcg_bars = ax.bar(
        x + width,
        ndcg,
        width,
        label="nDCG@5",
    )

    ax.set_title(
        "NexaGraph Retrieval Evaluation",
        fontsize=16,
        pad=15,
    )

    ax.set_xlabel("Retrieval Method", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.set_ylim(0, 0.9)

    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    add_value_labels(ax, recall_bars, recall)
    add_value_labels(ax, mrr_bars, mrr)
    add_value_labels(ax, ndcg_bars, ndcg)

    add_benchmark_note(fig, report)

    fig.subplots_adjust(bottom=0.19, top=0.87)

    save_plot(fig, "retrieval_metrics_comparison.png")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    report = load_results()

    methods, recall, mrr, ndcg = extract_metrics(report)

    print("\nNEXAGRAPH RETRIEVAL EVALUATION")
    print("=" * 65)

    print(
        f"Eligible cases: {report.get('eligible_case_count', 'Unknown')}"
    )
    print(
        f"Total cases: {report.get('total_case_count', 'Unknown')}"
    )

    print("\nAGGREGATE METRICS")
    print("-" * 65)

    print(
        f"{'Method':<16}"
        f"{'Recall@5':>12}"
        f"{'MRR':>12}"
        f"{'nDCG@5':>12}"
    )

    for i, method in enumerate(methods):
        print(
            f"{method:<16}"
            f"{recall[i]:>11.4f}"
            f"{mrr[i]:>12.4f}"
            f"{ndcg[i]:>12.4f}"
        )

    print("\nGENERATING VISUALIZATIONS")
    print("-" * 65)

    plot_recall(methods, recall, report)
    plot_ranking_quality(methods, mrr, ndcg, report)
    plot_all_metrics(methods, recall, mrr, ndcg, report)

    print("\nSUCCESS: All three plots generated.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

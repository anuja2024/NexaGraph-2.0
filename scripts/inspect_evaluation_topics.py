
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CORPUS = ROOT / "data/staging/siemens_pdf/chunks.jsonl"
OUTPUT = ROOT / "data/staging/siemens_pdf/evaluation_topic_candidates_v2.txt"

TOPICS = {
    "corporate": {
        "revenue": ["revenue", "sales", "turnover"],
        "profitability": ["profit", "margin", "earnings"],
        "business_segments": [
            "digital industries",
            "smart infrastructure",
            "mobility",
        ],
        "cash_flow": ["cash flow", "free cash flow"],
        "research_development": [
            "research and development",
            "R&D",
            "innovation",
        ],
    },
    "sustainability": {
        "climate": [
            "climate change",
            "greenhouse gas",
            "emissions",
        ],
        "energy": [
            "energy consumption",
            "renewable energy",
        ],
        "decarbonization": [
            "decarbonization",
            "carbon neutrality",
        ],
        "water": [
            "water consumption",
            "water withdrawal",
        ],
        "pollution": ["pollution", "waste"],
        "biodiversity": ["biodiversity", "ecosystem"],
    },
    "industrial_automation": {
        "cybersecurity": ["cybersecurity", "cyber threats"],
        "network_security": ["network security", "firewall"],
        "access_control": ["access control", "authentication"],
        "security_updates": [
            "security update",
            "patch management",
        ],
        "industrial_systems": [
            "SIMATIC",
            "process automation",
        ],
    },
}

TOP_N = 8
MIN_TEXT_LENGTH = 180
MAX_TEXT_LENGTH = 800

NOISY_TERMS = [
    "table of contents",
    "contents",
]


def load_chunks():
    if not CORPUS.exists():
        raise FileNotFoundError(CORPUS)

    chunks = []

    with CORPUS.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

    return chunks


def score_chunk(text, keywords):
    normalized = re.sub(r"\s+", " ", text.lower())

    matched = [
        keyword
        for keyword in keywords
        if keyword.lower() in normalized
    ]

    if not matched:
        return 0, []

    score = len(matched) * 10

    # Prefer substantive evidence passages.
    if len(text) >= MIN_TEXT_LENGTH:
        score += 3

    # Penalize likely table-of-contents passages.
    if any(term in normalized for term in NOISY_TERMS):
        score -= 15

    # Reward multiple distinct keyword occurrences.
    score += min(
        sum(normalized.count(k.lower()) for k in matched),
        10,
    )

    return score, matched


def discover(chunks):
    results = defaultdict(list)

    for chunk in chunks:
        project = chunk.get("project_id")
        text = chunk.get("text", "")

        if not isinstance(text, str):
            continue

        if len(text.strip()) < MIN_TEXT_LENGTH:
            continue

        for topic, keywords in TOPICS.get(project, {}).items():
            score, matched = score_chunk(text, keywords)

            if score <= 0:
                continue

            results[(project, topic)].append({
                "score": score,
                "page": chunk.get("page_number"),
                "chunk": chunk.get("chunk_index"),
                "matched_keywords": matched,
                "text": re.sub(
                    r"\s+", " ", text
                )[:MAX_TEXT_LENGTH],
            })

    return results


def main():
    chunks = load_chunks()
    results = discover(chunks)

    lines = [
        "NEXAGRAPH EVALUATION TOPIC DISCOVERY V2",
        "=" * 75,
        f"Total corpus chunks: {len(chunks)}",
        f"Top candidates per topic: {TOP_N}",
        "",
        "NOTE: Scores are keyword-based ranking scores,",
        "not evidence verification or retrieval metrics.",
    ]

    for project, topics in TOPICS.items():
        lines.extend([
            "",
            "",
            f"PROJECT: {project.upper()}",
            "=" * 75,
        ])

        for topic in topics:
            candidates = sorted(
                results[(project, topic)],
                key=lambda item: item["score"],
                reverse=True,
            )[:TOP_N]

            lines.extend([
                "",
                f"TOPIC: {topic}",
                f"Candidates found: {len(results[(project, topic)])}",
            ])

            for item in candidates:
                lines.extend([
                    "",
                    f"Score: {item['score']}",
                    f"Page: {item['page']} | Chunk: {item['chunk']}",
                    f"Keywords: {', '.join(item['matched_keywords'])}",
                    f"Evidence: {item['text']}",
                ])

    OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Evidence discovery V2 completed.")
    print("Chunks inspected:", len(chunks))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()

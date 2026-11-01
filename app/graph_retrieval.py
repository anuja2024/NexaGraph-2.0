
from functools import lru_cache
from pathlib import Path
from typing import Any
import logging
import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "can", "could", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "if", "in", "into", "is", "it",
    "its", "may", "might", "of", "on", "or", "our", "should",
    "that", "the", "their", "them", "there", "these", "they",
    "this", "those", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "you", "your",
    "please", "tell", "show", "give", "use", "using",
}

SCENARIO_CUES = (
    "pro forma",
    "pro-forma",
    "hypothetical",
    "hypothetically",
    "would have been",
    "had been included",
    "if included",
    "if it had",
    "standalone",
    "stand-alone",
    "not included",
    "excluding the effects",
)


def tokenize(text: str) -> list[str]:
    """Return unique, informative lowercase query terms."""
    raw_terms = re.findall(r"\w+", text.lower())
    return list(dict.fromkeys(
        term for term in raw_terms
        if term not in STOP_WORDS and len(term) > 1
    ))


def chunk_key(record: dict[str, Any]) -> tuple:
    """Stable identity used to reconcile graph results with authorized chunks."""
    return (
        record.get("project_id"),
        record.get("document_id"),
        record.get("source_file"),
        record.get("page_number"),
        record.get("chunk_index"),
        record.get("text", ""),
    )


@lru_cache(maxsize=1)
def get_driver():
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all((uri, username, password)):
        raise RuntimeError("Neo4j configuration is unavailable")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    return driver


def _relevance_score(
    query: str,
    record: dict[str, Any],
) -> float:
    """
    Generic query-aware evidence score.

    Rewards query-term coverage and relevant multiword phrases.
    When a query explicitly excludes alternative scenarios, penalizes
    passages that describe those scenarios.
    """
    text = re.sub(r"\s+", " ", str(record.get("text") or "")).lower()
    query_lower = re.sub(r"\s+", " ", query.lower())
    terms = tokenize(query)

    if not text or not terms:
        return 0.0

    present_terms = sum(
        1 for term in terms
        if re.search(r"\b" + re.escape(term) + r"\b", text)
    )
    coverage = present_terms / max(len(terms), 1)

    # Reward phrases that occur together in the query and passage.
    query_phrases = []
    for match in re.finditer(r'"([^"]{3,})"', query_lower):
        query_phrases.append(match.group(1).strip())

    phrase_score = sum(
        2.0 for phrase in query_phrases if phrase in text
    )

    # Reward adjacent pairs of informative query terms when present.
    for left, right in zip(terms, terms[1:]):
        phrase = f"{left} {right}"
        if phrase in text:
            phrase_score += 0.35

    score = coverage * 5.0 + phrase_score

    # Queries requesting multiple facts benefit from passages containing
    # evidence for more than one of the requested concepts.
    metric_groups = (
        {"revenue", "sales", "turnover"},
        {"profit", "income", "earnings", "loss"},
    )
    query_lower_terms = set(terms)
    requested_groups = [
        group for group in metric_groups
        if query_lower_terms.intersection(group)
    ]
    covered_groups = sum(
        1 for group in requested_groups
        if any(
            re.search(r"\b" + re.escape(term) + r"\b", text)
            for term in group
        )
    )
    if len(requested_groups) > 1:
        score += 1.5 * covered_groups
        if covered_groups == len(requested_groups):
            score += 1.5

    # Apply scenario penalties only when the user explicitly asks to exclude
    # or distinguish alternative scenarios.
    exclusion_requested = bool(re.search(
        r"\b(exclude|excluding|excluding|not|rather than|instead of|"
        r"actual|reported|avoid|without)\b",
        query_lower,
    ))

    if exclusion_requested:
        scenario_matches = sum(
            1 for cue in SCENARIO_CUES if cue in text
        )
        score -= 2.5 * scenario_matches

        # Penalize conditional hypothetical formulations.
        if re.search(r"\bif\b.{0,100}\bwould have been\b", text):
            score -= 2.0

    return score


def retrieve_graph_candidates(
    project_id: str,
    query: str,
    authorized_chunks: list[dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieve graph-linked chunks, then rerank them against the query.

    Graph results are reconciled with authorized_chunks before being returned.
    """
    terms = tokenize(query)
    if not terms or not authorized_chunks or limit < 1:
        return []

    scoped = [
        chunk
        for chunk in authorized_chunks
        if chunk.get("project_id") == project_id
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]
    if not scoped:
        return []

    authorized_keys = {chunk_key(chunk) for chunk in scoped}
    authorized_document_ids = {
        chunk.get("document_id") for chunk in scoped
        if chunk.get("document_id")
    }

    with get_driver().session() as session:
        records = session.run(
            """
            MATCH (p:Project {project_id: $project_id})
                  -[:IN_PROJECT]-(d:Document)
                  -[:HAS_CHUNK]-(c:Chunk)
            WHERE d.project_id = $project_id
              AND c.project_id = $project_id
              AND d.document_id IN $authorized_document_ids
              AND c.document_id IN $authorized_document_ids
            WITH c,
                 [term IN $terms
                  WHERE toLower(coalesce(c.text, ''))
                        CONTAINS term] AS matched
            WHERE size(matched) > 0
            RETURN c.project_id AS project_id,
                   c.document_id AS document_id,
                   c.source_file AS source_file,
                   c.page_number AS page_number,
                   c.chunk_index AS chunk_index,
                   c.text AS text,
                   size(matched) AS graph_match_count,
                   matched AS graph_matched_terms
            ORDER BY graph_match_count DESC, page_number ASC
            LIMIT $candidate_limit
            """,
            project_id=project_id,
            authorized_document_ids=sorted(authorized_document_ids),
            terms=terms,
            candidate_limit=min(max(limit * 10, 50), 100),
        ).data()

    results = []
    seen = set()

    for record in records:
        # Keep the authorization identity check as a hard requirement.
        key = chunk_key(record)
        if key not in authorized_keys or key in seen:
            continue
        seen.add(key)

        item = {
            "project_id": record["project_id"],
            "document_id": record["document_id"],
            "source_file": record["source_file"],
            "page_number": record["page_number"],
            "chunk_index": record["chunk_index"],
            "text": record["text"],
            "graph_score": float(record["graph_match_count"]),
            "graph_matched_terms": record["graph_matched_terms"],
        }
        item["relevance_score"] = _relevance_score(query, item)
        results.append(item)

    results.sort(
        key=lambda item: (
            item["relevance_score"],
            item["graph_score"],
        ),
        reverse=True,
    )
    return results[:limit]


def retrieve_graph_enhanced(
    project_id: str,
    query: str,
    authorized_chunks: list[dict[str, Any]],
    limit: int = 5,
) -> dict[str, Any]:
    """Fuse hybrid and graph rankings, then rerank the combined candidates."""
    from app.hybrid_retrieval import retrieve_hybrid

    if limit < 1:
        raise ValueError("limit must be at least 1")

    hybrid = retrieve_hybrid(
        project_id=project_id,
        query=query,
        authorized_chunks=authorized_chunks,
        limit=max(limit, 10),
        candidate_limit=30,
    )
    hybrid_results = hybrid.get("results", [])

    # Request a wider graph pool so useful passages aren't discarded
    # before the combined ranking is calculated.
    graph_results = retrieve_graph_candidates(
        project_id=project_id,
        query=query,
        authorized_chunks=authorized_chunks,
        limit=50,
    )

    hybrid_diagnostics = dict(
        hybrid.get("retrieval_diagnostics") or {}
    )

    if not graph_results:
        # Preserve the hybrid retriever's actual method, including fallbacks.
        hybrid_diagnostics.update({
            "graph_candidate_count": 0,
            "combined_candidate_count": len(hybrid_results),
        })
        return {
            "retrieval_method": hybrid.get(
                "retrieval_method", "hybrid_rrf"
            ),
            "results": hybrid_results[:limit],
            "retrieval_diagnostics": hybrid_diagnostics,
        }

    rankings = [
        ("hybrid", hybrid_results),
        ("graph", graph_results),
    ]
    fused: dict[tuple, dict[str, Any]] = {}

    for source_name, ranking in rankings:
        for rank, record in enumerate(ranking, start=1):
            key = chunk_key(record)
            if key not in fused:
                fused[key] = {
                    "record": record,
                    "rrf_score": 0.0,
                    "hybrid_rank": None,
                    "graph_rank": None,
                }

            fused[key]["rrf_score"] += 1.0 / (60 + rank)
            fused[key][f"{source_name}_rank"] = rank

    ordered = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    # Rerank the complete fused candidate set instead of truncating it first.
    for item in ordered:
        record = dict(item["record"])
        item["record"] = record
        item["relevance_score"] = _relevance_score(query, record)

        # Relevance is the primary criterion; RRF breaks close ties.
        item["final_score"] = (
            item["relevance_score"] + item["rrf_score"]
        )

    ordered.sort(
        key=lambda item: (
            item["final_score"],
            item["rrf_score"],
        ),
        reverse=True,
    )

    results = []
    for item in ordered[:limit]:
        record = dict(item["record"])
        record["score"] = float(item["final_score"])
        record["rrf_score"] = float(item["rrf_score"])
        record["relevance_score"] = float(item["relevance_score"])
        record["hybrid_rank"] = item["hybrid_rank"]
        record["graph_rank"] = item["graph_rank"]
        record.pop("graph_score", None)
        record.pop("graph_matched_terms", None)
        record.pop("matched_terms", None)
        record.pop("_bm25_score", None)
        record.pop("_semantic_score", None)
        results.append(record)

    hybrid_method = hybrid.get("retrieval_method", "hybrid_rrf")
    if hybrid_method == "BM25_fallback":
        method = "BM25_graph_reranked"
    else:
        method = "hybrid_graph_reranked"

    hybrid_diagnostics.update({
        "graph_candidate_count": len(graph_results),
        "combined_candidate_count": len(fused),
    })

    return {
        "retrieval_method": method,
        "results": results,
        "retrieval_diagnostics": hybrid_diagnostics,
    }

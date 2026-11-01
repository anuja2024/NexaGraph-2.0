import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastembed import TextEmbedding
from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = PROJECT_ROOT / "data" / "vector_store" / "qdrant_local"

COLLECTION_NAME = "nexagraph_siemens_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

DEFAULT_CANDIDATE_LIMIT = 10
DEFAULT_RESULT_LIMIT = 5
RRF_K = 60
MAX_SEQUENCE_LENGTH = 256


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def chunk_key(record: dict[str, Any]) -> tuple:
    return (
        record.get("project_id"),
        record.get("document_id"),
        record.get("source_file"),
        record.get("page_number"),
        record.get("chunk_index"),
        record.get("text", ""),
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    if not QDRANT_PATH.exists():
        raise RuntimeError("Local Qdrant index is unavailable")

    client = QdrantClient(path=str(QDRANT_PATH))
    if not client.collection_exists(COLLECTION_NAME):
        client.close()
        raise RuntimeError("NexaGraph Qdrant collection is unavailable")

    return client


def close_qdrant_client() -> None:
    """Close the cached local Qdrant client during application shutdown."""
    if get_qdrant_client.cache_info().currsize == 0:
        return

    client = get_qdrant_client()
    get_qdrant_client.cache_clear()
    client.close()

@lru_cache(maxsize=1)
def get_reranker() -> tuple[Tokenizer, ort.InferenceSession]:
    model_path = hf_hub_download(
        repo_id=RERANKER_MODEL,
        filename="onnx/model.onnx",
        local_files_only=True,
    )
    tokenizer_path = hf_hub_download(
        repo_id=RERANKER_MODEL,
        filename="tokenizer.json",
        local_files_only=True,
    )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    return tokenizer, session


def bm25_search(
    authorized_chunks: list[dict[str, Any]],
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    chunks = []
    tokenized = []

    for chunk in authorized_chunks:
        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        terms = tokenize(text)
        if terms:
            chunks.append(chunk)
            tokenized.append(terms)

    if not chunks:
        return []

    model = BM25Okapi(tokenized)
    scores = model.get_scores(query_tokens)
    query_terms = set(query_tokens)
    ranked = []

    for index, score in enumerate(scores):
        matched = sorted(query_terms.intersection(tokenized[index]))
        if not matched:
            continue

        record = dict(chunks[index])
        record["_bm25_score"] = float(score)
        record["_matched_terms"] = matched
        ranked.append(record)

    ranked.sort(key=lambda item: item["_bm25_score"], reverse=True)
    return ranked[:limit]


def semantic_search(
    project_id: str,
    query: str,
    limit: int,
    authorized_keys: set[tuple],
) -> list[dict[str, Any]]:
    if not authorized_keys:
        return []

    embedding_model = get_embedding_model()
    vectors = list(embedding_model.query_embed(query))
    if len(vectors) != 1:
        raise RuntimeError("Expected exactly one query embedding")

    response = get_qdrant_client().query_points(
        collection_name=COLLECTION_NAME,
        query=[float(value) for value in vectors[0]],
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="project_id",
                    match=MatchValue(value=project_id),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )

    results = []
    for point in response.points:
        payload = dict(point.payload or {})
        payload["_semantic_score"] = float(point.score)

        # Reconcile every vector result with the authorized corpus.
        # This prevents Qdrant payloads alone from granting access.
        if chunk_key(payload) in authorized_keys:
            results.append(payload)

    return results


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    fused: dict[tuple, dict[str, Any]] = {}

    for name, ranking in zip(("bm25", "semantic"), rankings):
        for rank, record in enumerate(ranking, start=1):
            key = chunk_key(record)

            if key not in fused:
                fused[key] = {
                    "record": record,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "semantic_rank": None,
                }

            fused[key]["rrf_score"] += 1.0 / (RRF_K + rank)
            fused[key][f"{name}_rank"] = rank

            if name == "bm25":
                fused[key]["record"] = record

    return sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def query_aware_adjustment(query: str, text: str) -> float:
    """Adjust reranker scores for explicit query constraints."""
    import re

    q = re.sub(r"\s+", " ", query.lower())
    passage = re.sub(r"\s+", " ", text.lower())
    score = 0.0

    # Penalize hypothetical scenarios only when the question excludes them.
    exclusion_requested = bool(re.search(
        r"\b(exclude|excluding|not|rather than|instead of|actual|"
        r"reported|avoid|without)\b",
        q,
    ))

    if exclusion_requested:
        cues = (
            "pro forma", "pro-forma", "hypothetical",
            "would have been", "had been included",
            "if included", "if it had",
        )
        score -= 3.0 * sum(cue in passage for cue in cues)
        if re.search(r"\bif\b.{0,140}\bwould have been\b", passage):
            score -= 2.0

    # Prefer group-level evidence when the question explicitly requests it.
    group_requested = bool(re.search(
        r"\b(consolidated|group|group-level)\b", q
    ))
    if group_requested:
        if re.search(
            r"\b(consolidated|siemens group|group level)\b", passage
        ):
            score += 1.5
        if re.search(
            r"\bstandalone\b|\bstand-alone\b|"
            r"annual financial statements of siemens ag",
            passage,
        ):
            score -= 2.5

    # Prefer evidence that covers all requested metric categories.
    wants_revenue = bool(re.search(r"\b(revenue|sales|turnover)\b", q))
    wants_income = bool(re.search(
        r"\b(net income|income|profit|earnings|loss)\b", q
    ))
    has_revenue = bool(re.search(
        r"\b(revenue|sales|turnover)\b", passage
    ))
    has_income = bool(re.search(
        r"\b(net income|income|profit|earnings|loss)\b", passage
    ))

    if wants_revenue and has_revenue:
        score += 1.0
    if wants_income and has_income:
        score += 1.0
    if wants_revenue and wants_income and has_revenue and has_income:
        score += 1.5

    return score


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rerank with the cross-encoder and query-aware adjustments."""
    tokenizer, session = get_reranker()
    input_names = {item.name for item in session.get_inputs()}
    scored = []

    for candidate in candidates:
        record = candidate["record"]
        encoded = tokenizer.encode(query, record.get("text", ""))

        inputs = {
            "input_ids": np.asarray([encoded.ids], dtype=np.int64),
            "attention_mask": np.asarray(
                [encoded.attention_mask], dtype=np.int64
            ),
        }

        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.asarray(
                [encoded.type_ids], dtype=np.int64
            )

        output = session.run(None, inputs)[0]
        raw_score = float(np.asarray(output).reshape(-1)[0])
        adjustment = query_aware_adjustment(
            query, str(record.get("text") or "")
        )

        item = dict(candidate)
        item["reranker_score"] = raw_score
        item["query_adjustment"] = adjustment
        item["final_reranker_score"] = raw_score + adjustment
        scored.append(item)

    scored.sort(
        key=lambda item: item["final_reranker_score"],
        reverse=True,
    )
    return scored




def retrieve_hybrid(
    project_id: str,
    query: str,
    authorized_chunks: list[dict[str, Any]],
    limit: int = DEFAULT_RESULT_LIMIT,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> dict[str, Any]:
    """Retrieve only from caller-authorized chunks using hybrid ranking."""
    if not query.strip() or not authorized_chunks:
        return {
            "retrieval_method": "hybrid_rrf",
            "results": [],
            "retrieval_diagnostics": {
                "authorized_chunk_count": len(authorized_chunks),
                "bm25_candidate_count": 0,
                "semantic_candidate_count": 0,
                "fused_candidate_count": 0,
                "cross_encoder_used": False,
                "cross_encoder_status": "not_run",
                "semantic_status": "not_run",
                "semantic_fallback": False,
                "reranker_fallback": False,
            },
        }

    if limit < 1 or candidate_limit < limit:
        raise ValueError("Invalid retrieval limits")

    scoped_chunks = [
        chunk
        for chunk in authorized_chunks
        if chunk.get("project_id") == project_id
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]

    if not scoped_chunks:
        return {
            "retrieval_method": "hybrid_rrf",
            "results": [],
            "retrieval_diagnostics": {
                "authorized_chunk_count": 0,
                "bm25_candidate_count": 0,
                "semantic_candidate_count": 0,
                "fused_candidate_count": 0,
                "cross_encoder_used": False,
                "cross_encoder_status": "not_run",
                "semantic_status": "not_run",
                "semantic_fallback": False,
                "reranker_fallback": False,
            },
        }

    authorized_keys = {chunk_key(chunk) for chunk in scoped_chunks}
    lexical = bm25_search(scoped_chunks, query, candidate_limit)

    diagnostics = {
        "authorized_chunk_count": len(scoped_chunks),
        "bm25_candidate_count": len(lexical),
        "semantic_candidate_count": 0,
        "fused_candidate_count": 0,
        "cross_encoder_used": False,
        "cross_encoder_status": "not_run",
        "semantic_status": "not_run",
        "semantic_fallback": False,
        "reranker_fallback": False,
    }

    # If semantic retrieval is unavailable, don't fail the whole API.
    # Keep the fallback explicit in both the method and diagnostics.
    try:
        semantic = semantic_search(
            project_id,
            query,
            candidate_limit,
            authorized_keys,
        )
        diagnostics["semantic_candidate_count"] = len(semantic)
        diagnostics["semantic_status"] = "success"
    except Exception:
        logger.exception("Semantic retrieval failed; using BM25 fallback")
        diagnostics["semantic_status"] = "failed"
        diagnostics["semantic_fallback"] = True
        results = []
        for record in lexical[:limit]:
            item = dict(record)
            item["score"] = float(item.pop("_bm25_score", 0.0))
            item["matched_terms"] = item.pop("_matched_terms", [])
            results.append(item)
        return {
            "retrieval_method": "BM25_fallback",
            "results": results,
            "retrieval_diagnostics": diagnostics,
        }

    fused = reciprocal_rank_fusion([lexical, semantic])
    diagnostics["fused_candidate_count"] = len(fused)

    if not fused:
        return {
            "retrieval_method": "hybrid_rrf",
            "results": [],
            "retrieval_diagnostics": diagnostics,
        }

    try:
        reranked = rerank_candidates(query, fused[:candidate_limit])
        method = "hybrid_rrf_cross_encoder"
        diagnostics["cross_encoder_used"] = True
        diagnostics["cross_encoder_status"] = "success"
    except Exception:
        logger.exception("Cross-encoder unavailable; using RRF ranking")
        reranked = fused
        method = "hybrid_rrf"
        diagnostics["cross_encoder_status"] = "failed"
        diagnostics["reranker_fallback"] = True

    results = []
    for item in reranked[:limit]:
        record = dict(item["record"])
        record["score"] = float(
            item.get(
                "final_reranker_score",
                item.get("reranker_score", item["rrf_score"]),
            )
        )
        record["rrf_score"] = float(item["rrf_score"])
        record["bm25_rank"] = item["bm25_rank"]
        record["semantic_rank"] = item["semantic_rank"]

        if "reranker_score" in item:
            record["reranker_score"] = float(item["reranker_score"])

        record["matched_terms"] = record.pop("_matched_terms", [])
        record.pop("_bm25_score", None)
        record.pop("_semantic_score", None)
        results.append(record)

    return {
        "retrieval_method": method,
        "results": results,
        "retrieval_diagnostics": diagnostics,
    }

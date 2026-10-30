"""
Hybrid retrieval pipeline:
  1. BM25 sparse search  (rank-bm25)
  2. Dense vector search  (Qdrant + BGE-small)
  3. Reciprocal Rank Fusion  (merges both result lists)
  4. Cross-encoder reranking  (ms-marco-MiniLM — local, free)
"""

import logging
import re
from functools import lru_cache

from sentence_transformers import CrossEncoder
from qdrant_client.models import Filter

from ..core.config import settings
from ..core.embeddings import embed_query
from .indexer import get_qdrant, get_bm25, get_all_chunks_by_ids

logger = logging.getLogger(__name__)


# ── Cross-encoder ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    """Cached cross-encoder reranker — runs locally, free."""
    logger.info("Initializing cross-encoder reranker %s", settings.reranker_model)
    return CrossEncoder(settings.reranker_model)


# ── Tokenizer (mirrors indexer) ───────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


# ── BM25 search ───────────────────────────────────────────────────────────────

def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    bm25, docs = get_bm25()
    if bm25 is None or not docs:
        logger.warning("BM25 search requested before index was loaded")
        return []

    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)

    # Pair each doc with its BM25 score and sort
    scored = sorted(
        [(docs[i], float(scores[i])) for i in range(len(docs))],
        key=lambda x: x[1],
        reverse=True,
    )
    results = [
        {**doc, "score": score, "rank_source": "bm25"}
        for doc, score in scored[:top_k]
        if score > 0
    ]
    logger.info("BM25 search returned %d candidates", len(results))
    return results


# ── Dense vector search ───────────────────────────────────────────────────────

def vector_search(query: str, top_k: int = 20, source_filter: str | None = None) -> list[dict]:
    client = get_qdrant()
    logger.info("Vector search started (query_chars=%d, top_k=%d, source_filter=%s)", len(query), top_k, source_filter)
    query_vector = embed_query(query)

    qdrant_filter = None
    if source_filter:
        from qdrant_client.models import FieldCondition, MatchValue
        qdrant_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    hits = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
        with_payload=True,
    )

    results = [
        {
            "id": str(hit.id),
            "content": hit.payload["content"],
            "source": hit.payload["source"],
            "url": hit.payload["url"],
            "title": hit.payload["title"],
            "score": float(hit.score),
            "rank_source": "vector",
        }
        for hit in hits
    ]
    logger.info("Vector search returned %d candidates", len(results))
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """Merge multiple ranked lists into one using RRF scoring."""
    scores: dict[str, float] = {}
    docs_by_id: dict[str, dict] = {}

    for ranked_list in result_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            docs_by_id[doc_id] = doc  # last writer wins for payload

    merged = sorted(docs_by_id.values(), key=lambda d: scores[d["id"]], reverse=True)
    for doc in merged:
        doc["score"] = round(scores[doc["id"]], 5)
    return merged


# ── Main hybrid search ────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    top_k: int = 5,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Full pipeline:
      BM25 + Vector → RRF fusion → Cross-encoder reranking → top_k results.
    """
    logger.info("Hybrid search started (query_chars=%d, top_k=%d, source_filter=%s)", len(query), top_k, source_filter)
    # Step 1 & 2: parallel sparse + dense
    bm25_results = bm25_search(query, top_k=20)
    vec_results = vector_search(query, top_k=20, source_filter=source_filter)

    # Apply source filter to BM25 results too (BM25 has no native filter)
    if source_filter:
        bm25_results = [r for r in bm25_results if r.get("source") == source_filter]

    # Step 3: RRF
    candidates = reciprocal_rank_fusion([bm25_results, vec_results])[:25]

    if not candidates:
        logger.warning("Hybrid search produced no candidates")
        return []

    # Step 4: Cross-encoder reranking
    ce = get_cross_encoder()
    pairs = [(query, c["content"]) for c in candidates]
    ce_scores = ce.predict(pairs).tolist()

    for i, c in enumerate(candidates):
        c["ce_score"] = round(float(ce_scores[i]), 4)

    reranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
    logger.info("Hybrid search completed with %d reranked results", len(reranked[:top_k]))
    return reranked[:top_k]

import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models.schemas import (
    AskRequest,
    IndexRequest,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    SourceChunk,
)
from ..services.indexer import DOC_SOURCES, index_source
from ..services.retrieval import hybrid_search
from ..services.llm import stream_answer

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/sources")
async def list_sources():
    """Return the available documentation sources that can be indexed."""
    sources = list(DOC_SOURCES.keys())
    logger.info("Listed %d StackBot doc sources", len(sources))
    return {"sources": sources}


@router.post("/index", response_model=IndexResponse)
async def index_docs(req: IndexRequest):
    """Trigger indexing for one or more documentation sources."""
    results: dict[str, int] = {}
    errors: list[str] = []

    logger.info("Received index request for %d sources", len(req.sources))

    for src in req.sources:
        try:
            n = index_source(src)
            results[src] = n
        except ValueError as e:
            logger.warning("Invalid indexing source requested: %s", src)
            errors.append(str(e))
        except Exception as e:
            logger.exception("Indexing failed for source %s", src)
            errors.append(f"{src}: {e}")

    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    logger.info("Indexed %d total chunks across %d sources", sum(results.values()), len(results))

    return IndexResponse(
        indexed=results,
        message=f"Indexed {sum(results.values())} chunks across {len(results)} source(s).",
    )


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Hybrid BM25 + vector search with cross-encoder reranking."""
    results = hybrid_search(req.query, top_k=req.top_k)
    logger.info("Search returned %d results for query_chars=%d", len(results), len(req.query))
    return SearchResponse(
        query=req.query,
        results=[
            SourceChunk(
                id=r["id"],
                content=r["content"],
                source=r["source"],
                url=r["url"],
                title=r["title"],
                score=r.get("score", 0.0),
                ce_score=r.get("ce_score"),
            )
            for r in results
        ],
    )


@router.post("/ask")
async def ask(req: AskRequest):
    """
    Streaming endpoint: hybrid search → cross-encoder rerank → Groq LLM.
    Returns Server-Sent Events so the frontend can stream tokens.
    """
    logger.info("Received ask request (query_chars=%d, top_k=%d)", len(req.query), req.top_k)
    chunks = hybrid_search(req.query, top_k=req.top_k)

    if not chunks:
        logger.warning("No documentation chunks found for query_chars=%d", len(req.query))
        async def no_results():
            yield "data: " + json.dumps({"token": "⚠️ No relevant documentation found. Try rephrasing or index more sources."}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_results(), media_type="text/event-stream")

    # Emit sources first as a metadata event
    sources_payload = [
        {"source": c["source"], "url": c["url"], "title": c["title"], "ce_score": c.get("ce_score", 0)}
        for c in chunks
    ]

    async def event_stream():
        yield "data: " + json.dumps({"type": "sources", "sources": sources_payload}) + "\n\n"
        for token in stream_answer(req.query, chunks):
            yield "data: " + json.dumps({"type": "token", "token": token}) + "\n\n"
        yield "data: [DONE]\n\n"
        logger.info("Completed ask stream with %d source chunks", len(chunks))

    return StreamingResponse(event_stream(), media_type="text/event-stream")

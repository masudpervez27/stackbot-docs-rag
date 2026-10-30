"""
Indexer — scrapes documentation pages, chunks them, and stores in:
  • Qdrant Cloud  (dense vectors)
  • BM25 in-memory index (persisted as pickle for sparse retrieval)
  • SQLite         (chunk metadata for display)

Run once to populate:
    uv run python -m app.services.indexer
"""

import logging
import os
import pickle
import re
import sqlite3
import uuid
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from ..core.config import settings
from ..core.embeddings import embed_texts

logger = logging.getLogger(__name__)

# ── Doc sources ──────────────────────────────────────────────────────────────

DOC_SOURCES: dict[str, list[str]] = {
    "fastapi": [
        "https://fastapi.tiangolo.com/",
        "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "https://fastapi.tiangolo.com/tutorial/path-params/",
        "https://fastapi.tiangolo.com/tutorial/query-params/",
        "https://fastapi.tiangolo.com/tutorial/request-body/",
        "https://fastapi.tiangolo.com/tutorial/response-model/",
        "https://fastapi.tiangolo.com/tutorial/dependencies/",
        "https://fastapi.tiangolo.com/tutorial/bigger-applications/",
        "https://fastapi.tiangolo.com/tutorial/background-tasks/",
        "https://fastapi.tiangolo.com/tutorial/middleware/",
    ],
    "langchain": [
        "https://python.langchain.com/docs/introduction/",
        "https://python.langchain.com/docs/concepts/",
        "https://python.langchain.com/docs/how_to/",
        "https://python.langchain.com/docs/tutorials/llm_chain/",
        "https://python.langchain.com/docs/tutorials/rag/",
        "https://python.langchain.com/docs/tutorials/agents/",
    ],
    "react": [
        "https://react.dev/learn",
        "https://react.dev/learn/thinking-in-react",
        "https://react.dev/learn/describing-the-ui",
        "https://react.dev/learn/adding-interactivity",
        "https://react.dev/learn/managing-state",
        "https://react.dev/reference/react/useState",
        "https://react.dev/reference/react/useEffect",
    ],
}

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── Qdrant client ─────────────────────────────────────────────────────────────

_qdrant: QdrantClient | None = None


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        logger.info("Initializing Qdrant client for StackBot")
        _qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return _qdrant


def ensure_collection() -> None:
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in existing:
        logger.info("Creating Qdrant collection %s", settings.qdrant_collection)
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )


# ── SQLite ────────────────────────────────────────────────────────────────────

DB_PATH = Path("./data/chunks.db")


def get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            content TEXT,
            source TEXT,
            url TEXT,
            title TEXT
        )"""
    )
    conn.commit()
    return conn


# ── BM25 index ────────────────────────────────────────────────────────────────

_bm25: BM25Okapi | None = None
_bm25_docs: list[dict] | None = None  # parallel list to BM25 corpus


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def load_bm25_index() -> tuple[BM25Okapi | None, list[dict]]:
    global _bm25, _bm25_docs
    path = Path(settings.bm25_index_path)
    if path.exists():
        with open(path, "rb") as f:
            _bm25, _bm25_docs = pickle.load(f)
        logger.info("Loaded BM25 index from %s with %d documents", path, len(_bm25_docs or []))
    else:
        _bm25, _bm25_docs = None, []
        logger.info("BM25 index file %s does not exist yet", path)
    return _bm25, _bm25_docs


def save_bm25_index(bm25: BM25Okapi, docs: list[dict]) -> None:
    path = Path(settings.bm25_index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((bm25, docs), f)
    logger.info("Saved BM25 index to %s with %d documents", path, len(docs))


def get_bm25() -> tuple[BM25Okapi | None, list[dict]]:
    global _bm25, _bm25_docs
    if _bm25 is None:
        load_bm25_index()
    return _bm25, _bm25_docs or []


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_source(source: str) -> int:
    """Fetch, chunk, embed, and store all pages for a given source. Returns chunk count."""
    urls = DOC_SOURCES.get(source)
    if not urls:
        raise ValueError(f"Unknown source: {source}. Choose from {list(DOC_SOURCES)}")

    logger.info("Indexing documentation source %s with %d URLs", source, len(urls))
    ensure_collection()
    conn = get_db()
    all_chunks: list[dict] = []

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            loader.requests_kwargs = {"timeout": 10}
            docs = loader.load()
        except Exception as e:
            logger.warning("Failed to load %s: %s", url, e)
            continue

        for doc in docs:
            doc.metadata["source"] = source
            doc.metadata["url"] = url
            doc.metadata["title"] = doc.metadata.get("title", url)

        chunks = SPLITTER.split_documents(docs)
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            rec = {
                "id": chunk_id,
                "content": chunk.page_content,
                "source": source,
                "url": chunk.metadata.get("url", url),
                "title": chunk.metadata.get("title", url),
            }
            all_chunks.append(rec)
            conn.execute(
                "INSERT OR REPLACE INTO chunks(id, content, source, url, title) VALUES(?,?,?,?,?)",
                (rec["id"], rec["content"], rec["source"], rec["url"], rec["title"]),
            )

    conn.commit()
    conn.close()

    if not all_chunks:
        logger.warning("No chunks were indexed for source %s", source)
        return 0

    # Batch embed
    texts = [c["content"] for c in all_chunks]
    vectors = embed_texts(texts)

    # Upsert to Qdrant
    client = get_qdrant()
    points = [
        PointStruct(
            id=c["id"],
            vector=v,
            payload={"content": c["content"], "source": c["source"], "url": c["url"], "title": c["title"]},
        )
        for c, v in zip(all_chunks, vectors)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)

    # Rebuild BM25 index from all stored chunks
    _, existing_docs = get_bm25()
    # Remove old chunks from same source, add new ones
    existing_docs = [d for d in existing_docs if d["source"] != source]
    existing_docs.extend(all_chunks)
    corpus = [_tokenize(d["content"]) for d in existing_docs]
    new_bm25 = BM25Okapi(corpus)
    save_bm25_index(new_bm25, existing_docs)

    global _bm25, _bm25_docs
    _bm25, _bm25_docs = new_bm25, existing_docs

    logger.info("Indexed %d chunks for source %s", len(all_chunks), source)
    return len(all_chunks)


def get_all_chunks_by_ids(ids: list[str]) -> list[dict]:
    conn = get_db()
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    logger.info("Indexing documentation sources")
    for src in DOC_SOURCES:
        logger.info("Indexing %s", src)
        n = index_source(src)
        logger.info("Indexed %s into %d chunks", src, n)
    logger.info("Done indexing documentation sources")

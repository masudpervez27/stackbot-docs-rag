import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def _configure_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()

    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
    else:
        root_logger.setLevel(level)

    return logging.getLogger("stackbot.main")

from .api.routes import router
from .core.config import settings

logger = _configure_logging()

app = FastAPI(
    title="StackBot API",
    description="Hybrid BM25 + vector search over developer documentation with cross-encoder reranking.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Pre-load BM25 index and embedding model on startup."""
    from .services.indexer import load_bm25_index
    from .core.embeddings import get_embeddings
    logger.info("Starting StackBot API")
    load_bm25_index()
    get_embeddings()  # warm up the model
    logger.info("StackBot startup completed")

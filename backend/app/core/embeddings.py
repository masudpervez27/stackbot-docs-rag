from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from .config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Cached BGE-small embedding model — runs locally, free."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embeddings().embed_documents(texts)


def embed_query(query: str) -> list[float]:
    return get_embeddings().embed_query(query)

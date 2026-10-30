from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Qdrant
    qdrant_url: str = "https://your-cluster.qdrant.io"
    qdrant_api_key: str = ""
    qdrant_collection: str = "stackbot_docs"

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Embedding model (local, no API key needed)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Cross-encoder reranker (local, no API key needed)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # BM25 index persistence path
    bm25_index_path: str = "./data/bm25_index.pkl"

    # CORS
    frontend_origin: str = "http://localhost:5173"


settings = Settings()

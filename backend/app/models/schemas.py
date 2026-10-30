from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SourceChunk(BaseModel):
    id: str
    content: str
    source: str        # e.g. "fastapi", "react", "langchain"
    url: str
    title: str
    score: float
    ce_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SourceChunk]


class IndexRequest(BaseModel):
    sources: list[str]  # e.g. ["fastapi", "react", "langchain"]


class IndexResponse(BaseModel):
    indexed: dict[str, int]  # source → chunk count
    message: str

import os
import logging
from typing import Generator

from groq import Groq
from ..core.config import settings

_client: Groq | None = None
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are StackBot — a senior developer assistant specialising in FastAPI, React, and LangChain.

Answer the user's question using ONLY the documentation context provided below.

Rules:
- Be precise and practical. Prefer working code examples over long explanations.
- Format code with proper markdown code fences and language tags (```python, ```jsx, etc.).
- If the context does not contain enough information, say so clearly and suggest the relevant docs URL.
- Never invent API signatures or configuration options.
- Keep answers concise — bullet points and short paragraphs are preferred."""


def _get_client() -> Groq:
    global _client
    if _client is None:
        logger.info("Initializing Groq client for StackBot")
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def stream_answer(query: str, context_chunks: list[dict]) -> Generator[str, None, None]:
    """Yield streaming tokens from Groq, grounded in the retrieved context."""
    logger.info("Streaming StackBot answer with %d context chunks", len(context_chunks))
    context_text = "\n\n---\n\n".join(
        f"[{c['source'].upper()} — {c['title']}]\n{c['content']}"
        for c in context_chunks
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Documentation context:\n\n{context_text}\n\n"
                f"---\n\nDeveloper question: {query}"
            ),
        },
    ]

    stream = _get_client().chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=1024,
        temperature=0.05,   # near-deterministic for technical answers
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token

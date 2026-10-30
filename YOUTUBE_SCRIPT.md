# YOUTUBE SCRIPT — StackBot: Build a Hybrid RAG Developer Assistant

**Video title:** I Built an AI That Answers Coding Questions From Official Docs (Hybrid Search + Reranking)  
**Target length:** ~22 minutes  
**Thumbnail concept:** Split screen — Google-like search bar on left, highlighted code answer on right. Text overlay: "BM25 + Vector Search"

---

## [INTRO — 0:00–1:30]

*[Screen: StackBot running, user types "How do I add middleware in FastAPI?" — answer streams in with syntax-highlighted code and source badges]*

"What if you could ask a question to a senior developer who's read every page of the FastAPI, React, and LangChain documentation — and they answer instantly, with working code examples and links back to the original source?

That's what we're building today. It's called StackBot.

And what makes this project genuinely interesting from an AI engineering perspective is the retrieval pipeline underneath. Most RAG demos just do a vector similarity search. StackBot does something much better: **hybrid retrieval** — combining BM25 keyword search with dense vector search, then fusing the results with Reciprocal Rank Fusion, and finally reranking with a cross-encoder.

This is the exact approach used in production search systems at companies like Cohere, Vespa, and Weaviate.

Let me show you the full build."

---

## [WHAT IS RAG — AND WHY DOES RETRIEVAL QUALITY MATTER — 1:30–4:00]

*[Slide: RAG diagram — retrieve → augment → generate]*

"Quick RAG refresher. The idea is simple: instead of asking an LLM to answer from its training data — which may be outdated or hallucinated — we give it relevant passages at query time.

The quality of your LLM answers is directly determined by the quality of your retrieved passages. Garbage in, garbage out.

The naive approach is pure vector search — embed your query, find the nearest chunks by cosine similarity. This works great for semantic questions. But it fails badly for exact terms.

Watch what happens when I search for 'FieldCondition' — that's a Qdrant-specific class. A vector search might return vaguely similar docs about filtering, but it'll miss the exact docs that use the term 'FieldCondition'. BM25 keyword search would catch it immediately.

*[Slide: BM25 vs Vector — comparison table]*

That's why production systems use **hybrid search**: run both, then combine the results intelligently.

The second problem is **score calibration**. BM25 scores and cosine similarities live on completely different scales. How do you merge a BM25 score of 3.4 with a cosine similarity of 0.87?

The answer: Reciprocal Rank Fusion. We'll cover this in detail."

---

## [PROJECT ARCHITECTURE — 4:00–6:00]

*[Screen: README Mermaid diagram]*

"Here's the full architecture.

Two pipelines:

**Indexing pipeline** — runs once:
1. WebBaseLoader scrapes documentation URLs from FastAPI, React, and LangChain
2. RecursiveCharacterTextSplitter chunks into 600-token pieces with 80-token overlap
3. BGE-small-en-v1.5 (free HuggingFace model, runs locally) embeds each chunk
4. Chunks go to: Qdrant Cloud for dense vector search, BM25Okapi for sparse search, SQLite for metadata

**Query pipeline** — runs on every question:
1. The question is embedded with BGE-small
2. BM25 returns top-20 keyword matches
3. Qdrant returns top-20 semantic matches
4. RRF merges them into a unified ranked list of ~25 candidates
5. Cross-encoder ms-marco-MiniLM reranks those 25 into the final top-5
6. Groq Llama 3.3-70b generates a streaming, code-aware answer

Frontend: React + Vite + Tailwind + DaisyUI dark theme. Streaming via Server-Sent Events.

Total cost: $0."

---

## [SETUP WITH UV — 6:00–7:00]

*[Screen: Terminal]*

"All Python dependencies managed with UV — Astral's blazing-fast package manager.

```bash
cd stackbot-docs-rag/backend

# Install all dependencies in one command
uv sync

# That's it. No pip, no conda, no venv activate.
```

UV reads `pyproject.toml` and creates a `.venv` automatically. Significantly faster than pip, and the lockfile ensures reproducible installs across machines.

Let me show you our `pyproject.toml`..."

*[Screen: pyproject.toml — highlight key deps: fastapi, qdrant-client, rank-bm25, sentence-transformers, groq]*

---

## [EMBEDDING SERVICE — 7:00–8:30]

*[Screen: core/embeddings.py]*

"The embedding model is BGE-small-en-v1.5 from BAAI. It runs entirely on CPU, no API key needed, completely free.

We use `lru_cache(maxsize=1)` — the model loads once on startup, then every call reuses the same instance. This is the singleton pattern and it's critical for performance in a web server.

`normalize_embeddings=True` means all vectors have unit length — cosine similarity becomes a simple dot product, which is faster and numerically stable.

`embed_query()` uses a slightly different encoding than `embed_documents()` — BGE-small prepends an instruction prefix for queries. This is called asymmetric encoding and it improves retrieval accuracy."

---

## [INDEXING PIPELINE — 8:30–11:00]

*[Screen: services/indexer.py]*

"The indexer is the most important offline step. Let's walk through it.

First, we define our doc sources — a dictionary mapping source names to lists of URLs:

```python
DOC_SOURCES = {
    'fastapi': ['https://fastapi.tiangolo.com/', ...],
    'react':   ['https://react.dev/learn', ...],
    'langchain': [...],
}
```

For each URL, we use `WebBaseLoader` from LangChain Community. It fetches the HTML, parses it with BeautifulSoup, and extracts the text content.

Then `RecursiveCharacterTextSplitter` with `chunk_size=600`. This is smaller than most tutorials use — 600 tokens is about one code block plus some explanation. Smaller chunks = more precise retrieval.

After chunking, we:
1. Batch embed all chunks with BGE-small
2. Upsert to Qdrant as PointStructs with UUID IDs
3. Store in SQLite for metadata lookup
4. Rebuild the BM25Okapi index and pickle it to disk

The BM25 tokenizer strips punctuation and lowercases — `re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()`. Simple but effective.

Let me run the indexer now..."

*[Screen: Terminal — running `uv run python -m app.services.indexer`]*

"You'll see it loading each URL, chunking, embedding, upserting. Takes about 3-4 minutes. You only ever run this once — or when you want to update the docs."

---

## [RETRIEVAL PIPELINE — THE CORE ALGORITHM — 11:00–15:30]

*[Screen: services/retrieval.py]*

"This is the heart of StackBot. Let's go through each stage.

### Stage 1: BM25 Sparse Search

```python
def bm25_search(query: str, top_k: int = 20) -> list[dict]:
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    # sort and return top-k
```

BM25 — Best Match 25 — is the scoring function behind most search engines including Elasticsearch. For each term in the query, it considers term frequency in the document, inverse document frequency across the corpus, and document length normalization.

### Stage 2: Dense Vector Search

```python
def vector_search(query: str, top_k: int = 20) -> list[dict]:
    query_vector = embed_query(query)
    hits = qdrant_client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
    )
```

Qdrant uses HNSW (Hierarchical Navigable Small World) — an approximate nearest neighbor algorithm. Extremely fast even with millions of vectors.

### Stage 3: Reciprocal Rank Fusion

This is the clever part.

```python
def reciprocal_rank_fusion(result_lists, k=60):
    for ranked_list in result_lists:
        for rank, doc in enumerate(ranked_list):
            scores[doc['id']] += 1.0 / (k + rank + 1)
```

*[Slide: RRF formula]*

$$score_{RRF}(d) = \sum_{r \in R} \frac{1}{k + rank_r(d)}$$

Why does this work so well? Because it only looks at rank positions, not the actual scores. A BM25 score of 3.4 and a cosine score of 0.87 are incomparable — but rank 1 and rank 1 mean the same thing.

The constant `k=60` dampens the influence of very high ranks and prevents any single list from dominating.

A document ranked #1 in both lists gets: `1/61 + 1/61 = 0.033`. A document ranked #5 in both gets `1/65 + 1/65 = 0.031`. The difference is small — that's intentional. RRF gives credit to documents that consistently appear across methods, not just those that score highest in one.

### Stage 4: Cross-Encoder Reranking

```python
ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
pairs = [(query, candidate['content']) for candidate in candidates]
ce_scores = ce.predict(pairs)
```

Here's the key insight: bi-encoders (like BGE-small) encode query and document **independently** — that's what makes them fast enough for retrieval. But they lose the interaction signal between query and document.

Cross-encoders process them **jointly** — full attention over both. Dramatically better accuracy, but O(n) inference per query.

That's why we use a two-stage approach: fast bi-encoder for retrieval (20+20 candidates), slow cross-encoder for reranking (25 candidates). Best of both worlds."

---

## [LLM SERVICE — 15:30–17:00]

*[Screen: services/llm.py]*

"The LLM service is straightforward — Groq Llama 3.3-70b with streaming.

The system prompt is where the magic happens:

```
You are StackBot — a senior developer assistant...
Rules:
- Be precise. Prefer working code examples over long explanations.
- Format code with proper markdown code fences.
- If the context does not contain enough information, say so clearly.
- Never invent API signatures or configuration options.
```

That last rule is critical. We explicitly instruct the model not to hallucinate API details. Combined with retrieval grounding, this dramatically reduces hallucination.

We use `temperature=0.05` — near-zero. For technical documentation answers, we want deterministic, precise output. Not creativity.

The `stream_answer()` function yields tokens one by one. The FastAPI endpoint wraps this in Server-Sent Events."

---

## [API ENDPOINTS — 17:00–18:30]

*[Screen: api/routes.py]*

"Three endpoints:

`GET /api/sources` — lists available doc sources. Simple.

`POST /api/index` — triggers indexing. Call once, or whenever you want to refresh.

`POST /api/ask` — the main endpoint. Returns SSE.

The key pattern for SSE in FastAPI:

```python
async def event_stream():
    yield 'data: ' + json.dumps({'type': 'sources', 'sources': sources_payload}) + '\n\n'
    for token in stream_answer(query, chunks):
        yield 'data: ' + json.dumps({'type': 'token', 'token': token}) + '\n\n'
    yield 'data: [DONE]\n\n'

return StreamingResponse(event_stream(), media_type='text/event-stream')
```

We first emit a 'sources' event so the frontend can show the source chips immediately — before any answer tokens arrive. Then we stream tokens. The `[DONE]` sentinel tells the client the stream is complete."

---

## [REACT FRONTEND — 18:30–20:30]

*[Screen: src/lib/api.js]*

"On the frontend, we use the Fetch API with a ReadableStream to consume the SSE. Not EventSource — because EventSource only supports GET requests, and we need POST with a JSON body.

```javascript
const reader = response.body.getReader()
const decoder = new TextDecoder()
// read chunks, split on newlines, parse JSON
```

The `App.jsx` maintains a history array — each search adds an entry at the top. This gives users a scrollable history of their session.

*[Screen: components/AnswerCard.jsx]*

The AnswerCard uses `react-markdown` with `remark-gfm` for GitHub-flavored markdown, and `react-syntax-highlighter` with the `vscDarkPlus` theme for code blocks. This gives us proper syntax highlighting for Python, JSX, bash — whatever the LLM outputs.

*[Screen: components/SourceChip.jsx]*

Source chips link back to the original documentation page. Each chip shows:
- The source (FastAPI/React/LangChain) in a color-coded badge
- The page title (truncated)
- The cross-encoder score

The cross-encoder score is valuable feedback — it tells the user how relevant the retrieved passage actually is."

---

## [DEMO — 20:30–21:30]

*[Screen: running app]*

"Let me demo with a few queries:

1. 'How do I add CORS middleware in FastAPI?' — watch the source chips appear, then the answer streams in with a complete working code example.

2. 'What is useEffect and when should I use it?' — notice it cites the react.dev reference page.

3. 'How do I build a RAG chain in LangChain?' — a more complex question. It pulls from multiple LangChain docs pages and synthesises an answer.

And here's the interesting one — 'What is FieldCondition?' — that's a very specific Qdrant API class. Vector search alone might miss it. Hybrid search catches it via BM25."

---

## [DEPLOYMENT — 21:30–22:30]

*[Screen: Render + Vercel dashboards]*

"Backend to Render, frontend to Vercel — both free tiers.

Render free tier sleeps after 15 minutes. First cold start takes ~30 seconds because the embedding model loads. In a real production app you'd use a paid tier or a keep-alive ping.

Don't commit your `.env` file — use `.env.example` as a template and set real values in the Render/Vercel environment variable dashboards."

---

## [OUTRO — 22:30–23:00]

"StackBot demonstrates the full hybrid retrieval stack used in production AI systems:

- BM25 for lexical precision
- Dense vectors for semantic understanding  
- RRF for parameter-free fusion
- Cross-encoder reranking for accuracy
- Groq streaming for real-time UX

This is a portfolio project that shows you understand retrieval quality tuning — one of the highest-signal skills for AI Engineer roles.

If you found this useful, subscribe and check the description for the GitHub repo and all the free services used. Next video: PolicyPal — a multi-tenant RAG chatbot with conversation memory. See you there."

---

## Technical Summary Card (for video description)

```
Tools & Models:
• BGE-small-en-v1.5 — local CPU embeddings (HuggingFace, free)
• cross-encoder/ms-marco-MiniLM-L-6-v2 — local CPU reranker (HuggingFace, free)
• rank-bm25 — BM25Okapi sparse retrieval (Python library, free)
• Qdrant Cloud — vector store (free tier)
• Groq Llama-3.3-70b — LLM with streaming (free)
• FastAPI — Python web framework
• React + Vite + DaisyUI — frontend
• UV — Python package manager

Full source code: github.com/your-username/stackbot-docs-rag
```

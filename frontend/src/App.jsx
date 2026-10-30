import { useState } from 'react'
import SearchBar from './components/SearchBar'
import AnswerCard from './components/AnswerCard'
import { fetchAsk } from './lib/api'

export default function App() {
    const [history, setHistory] = useState([])   // [{query, answer, sources}]
    const [streaming, setStreaming] = useState(false)
    const [error, setError] = useState(null)

    const handleSearch = async (query) => {
        setError(null)
        setStreaming(true)

        // Add a new in-progress entry
        const entry = { query, answer: '', sources: [], done: false }
        setHistory((prev) => [entry, ...prev])

        await fetchAsk({
            query,
            topK: 5,
            onSources: (sources) => {
                setHistory((prev) => {
                    const next = [...prev]
                    next[0] = { ...next[0], sources }
                    return next
                })
            },
            onToken: (token) => {
                setHistory((prev) => {
                    const next = [...prev]
                    next[0] = { ...next[0], answer: next[0].answer + token }
                    return next
                })
            },
            onDone: () => {
                setStreaming(false)
                setHistory((prev) => {
                    const next = [...prev]
                    next[0] = { ...next[0], done: true }
                    return next
                })
            },
            onError: (err) => {
                setStreaming(false)
                setError(err.message)
            },
        })
    }

    return (
        <div data-theme="night" className="min-h-screen bg-base-100 text-base-content">
            {/* Header */}
            <header className="navbar bg-base-200 border-b border-base-300 px-4 lg:px-8">
                <div className="flex-1">
                    <span className="text-xl font-bold tracking-tight">
                        Stack<span className="text-primary">Bot</span>
                    </span>
                    <span className="ml-2 text-xs text-base-content/50 hidden sm:inline">
                        FastAPI · React · LangChain docs
                    </span>
                </div>
                <div className="flex-none">
                    <a
                        href="https://github.com/your-username/stackbot-docs-rag"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="btn btn-ghost btn-sm"
                    >
                        GitHub
                    </a>
                </div>
            </header>

            {/* Main */}
            <main className="max-w-3xl mx-auto px-4 py-10 flex flex-col gap-8">

                {/* Hero */}
                {history.length === 0 && (
                    <div className="text-center space-y-3 py-8">
                        <h1 className="text-4xl font-bold">
                            Ask your <span className="text-primary">dev docs</span>
                        </h1>
                        <p className="text-base-content/60 text-lg">
                            Hybrid BM25 + vector search with cross-encoder reranking, powered by Groq Llama 3.
                        </p>
                        <div className="flex flex-wrap justify-center gap-2 pt-2">
                            {[
                                'How do I add CORS in FastAPI?',
                                'What is useEffect in React?',
                                'How to build a RAG chain in LangChain?',
                            ].map((q) => (
                                <button
                                    key={q}
                                    onClick={() => handleSearch(q)}
                                    className="btn btn-outline btn-sm"
                                    disabled={streaming}
                                >
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Search bar */}
                <SearchBar onSearch={handleSearch} loading={streaming} />

                {/* Error */}
                {error && (
                    <div className="alert alert-error">
                        <span>{error}</span>
                    </div>
                )}

                {/* Answer history */}
                <div className="flex flex-col gap-6">
                    {history.map((item, idx) => (
                        <AnswerCard
                            key={idx}
                            query={item.query}
                            answer={item.answer}
                            sources={item.sources}
                            streaming={idx === 0 && streaming}
                        />
                    ))}
                </div>
            </main>
        </div>
    )
}

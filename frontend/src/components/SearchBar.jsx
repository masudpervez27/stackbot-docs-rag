import { useState, useRef, useEffect } from 'react'

export default function SearchBar({ onSearch, loading }) {
    const [query, setQuery] = useState('')
    const inputRef = useRef(null)

    // Focus on mount
    useEffect(() => { inputRef.current?.focus() }, [])

    // Keyboard shortcut: / to focus
    useEffect(() => {
        const handler = (e) => {
            if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
                e.preventDefault()
                inputRef.current?.focus()
            }
        }
        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [])

    const handleSubmit = (e) => {
        e.preventDefault()
        const trimmed = query.trim()
        if (!trimmed || loading) return
        onSearch(trimmed)
    }

    return (
        <form onSubmit={handleSubmit} className="w-full">
            <div className="join w-full">
                <input
                    ref={inputRef}
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder='Ask a dev question… e.g. "How do I add CORS in FastAPI?" (press / to focus)'
                    className="input input-bordered join-item flex-1 text-base focus:outline-none focus:border-primary"
                    disabled={loading}
                    maxLength={500}
                />
                <button
                    type="submit"
                    className={`btn btn-primary join-item ${loading ? 'loading loading-spinner' : ''}`}
                    disabled={loading || !query.trim()}
                >
                    {loading ? '' : 'Ask'}
                </button>
            </div>
        </form>
    )
}

const API_BASE = import.meta.env.VITE_API_URL || ''

const LOG_LEVELS = { debug: 10, info: 20, warn: 30, error: 40 }
const ACTIVE_LEVEL = LOG_LEVELS[(import.meta.env.VITE_LOG_LEVEL || (import.meta.env.DEV ? 'debug' : 'info')).toLowerCase()] || LOG_LEVELS.info

function log(level, message, context) {
    if ((LOG_LEVELS[level] || LOG_LEVELS.info) < ACTIVE_LEVEL) return
    const writer = console[level] || console.log
    if (context === undefined) writer(`[StackBot] ${message}`)
    else writer(`[StackBot] ${message}`, context)
}

/**
 * Streams the /api/ask endpoint using the Fetch Streams API.
 *
 * Calls:
 *  - onSources(sources[])  — once, with the retrieved doc chunks
 *  - onToken(token)        — many times, streaming LLM tokens
 *  - onDone()              — when stream ends
 *  - onError(err)          — on network/parse error
 */
export async function fetchAsk({ query, topK = 5, onSources, onToken, onDone, onError }) {
    try {
        log('info', 'Opening ask stream', { topK, queryChars: query.length })
        const response = await fetch(`${API_BASE}/api/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: topK }),
        })

        if (!response.ok) {
            const text = await response.text()
            throw new Error(`API error ${response.status}: ${text}`)
        }

        const reader = response.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        while (true) {
            const { value, done } = await reader.read()
            if (done) break

            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue
                const data = line.slice(6).trim()
                if (data === '[DONE]') {
                    onDone?.()
                    return
                }
                try {
                    const parsed = JSON.parse(data)
                    if (parsed.type === 'sources') onSources?.(parsed.sources)
                    else if (parsed.type === 'token') onToken?.(parsed.token)
                    // legacy plain token
                    else if (parsed.token) onToken?.(parsed.token)
                } catch {
                    log('warn', 'Failed to parse streamed StackBot event', { length: data.length })
                }
            }
        }
        onDone?.()
    } catch (err) {
        log('error', 'Ask stream failed', { message: err.message })
        onError?.(err)
    }
}

export async function fetchSources() {
    log('debug', 'Fetching indexable documentation sources')
    const res = await fetch(`${API_BASE}/api/sources`)
    if (!res.ok) {
        const text = await res.text()
        log('error', 'Fetch sources failed', { status: res.status, body: text })
        throw new Error(text)
    }
    return res.json()
}

export async function triggerIndex(sources) {
    log('info', 'Triggering source indexing', { sourceCount: sources.length })
    const res = await fetch(`${API_BASE}/api/index`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sources }),
    })
    if (!res.ok) {
        const text = await res.text()
        log('error', 'Trigger index failed', { status: res.status, body: text })
        throw new Error(text)
    }
    return res.json()
}

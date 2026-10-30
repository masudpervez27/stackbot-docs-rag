export default function SourceChip({ source }) {
    const colorMap = {
        fastapi: 'badge-success',
        react: 'badge-info',
        langchain: 'badge-warning',
    }
    const color = colorMap[source.source?.toLowerCase()] || 'badge-ghost'
    const score = source.ce_score != null ? (source.ce_score > 0 ? `+${source.ce_score.toFixed(2)}` : source.ce_score.toFixed(2)) : null

    return (
        <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className={`badge ${color} badge-outline gap-1 hover:opacity-80 transition-opacity cursor-pointer text-xs`}
            title={source.title}
        >
            <span className="font-semibold uppercase">{source.source}</span>
            <span className="opacity-70 truncate max-w-[120px]">{source.title?.slice(0, 30)}</span>
            {score && <span className="ml-1 opacity-60">{score}</span>}
        </a>
    )
}

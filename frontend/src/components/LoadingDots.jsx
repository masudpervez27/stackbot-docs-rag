export default function LoadingDots() {
    return (
        <span className="inline-flex items-center gap-1" aria-label="Loading">
            {[0, 1, 2].map((i) => (
                <span
                    key={i}
                    className="block w-2 h-2 rounded-full bg-primary animate-bounce"
                    style={{ animationDelay: `${i * 0.15}s` }}
                />
            ))}
        </span>
    )
}

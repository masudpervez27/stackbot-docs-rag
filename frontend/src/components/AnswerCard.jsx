import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import remarkGfm from 'remark-gfm'
import SourceChip from './SourceChip'
import LoadingDots from './LoadingDots'

export default function AnswerCard({ query, answer, sources, streaming }) {
    return (
        <div className="card bg-base-200 shadow-lg w-full">
            <div className="card-body gap-4">

                {/* Query echo */}
                <div className="flex items-start gap-3">
                    <div className="badge badge-primary badge-lg shrink-0">Q</div>
                    <p className="text-base font-medium">{query}</p>
                </div>

                <div className="divider my-0" />

                {/* Sources */}
                {sources.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                        {sources.map((s, i) => (
                            <SourceChip key={i} source={s} />
                        ))}
                    </div>
                )}

                {/* Answer stream */}
                <div className="prose prose-invert max-w-none text-sm">
                    {answer ? (
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                                code({ node, inline, className, children, ...props }) {
                                    const match = /language-(\w+)/.exec(className || '')
                                    return !inline && match ? (
                                        <SyntaxHighlighter
                                            style={vscDarkPlus}
                                            language={match[1]}
                                            PreTag="div"
                                            {...props}
                                        >
                                            {String(children).replace(/\n$/, '')}
                                        </SyntaxHighlighter>
                                    ) : (
                                        <code className="bg-base-300 px-1 py-0.5 rounded text-xs" {...props}>
                                            {children}
                                        </code>
                                    )
                                },
                            }}
                        >
                            {answer}
                        </ReactMarkdown>
                    ) : null}
                    {streaming && !answer && <LoadingDots />}
                    {streaming && answer && (
                        <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-0.5 align-bottom" />
                    )}
                </div>
            </div>
        </div>
    )
}

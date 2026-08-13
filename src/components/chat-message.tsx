'use client'

import React from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Copy, Check, RotateCcw } from 'lucide-react'

interface ChatMessageProps {
  content: string
  role: 'user' | 'assistant'
  isError?: boolean
  onRetry?: () => void
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = React.useState(false)
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  return (
    <button
      onClick={handleCopy}
      className="absolute top-2 right-2 p-1.5 rounded-md bg-white/10 hover:bg-white/20 transition-colors text-xs text-white/70 hover:text-white"
      title="Copy code"
    >
      {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
    </button>
  )
}

export function ChatMessageBubble({ content, role, isError, onRetry }: ChatMessageProps) {
  return (
    <div className={`rounded-2xl px-4 py-3 max-w-[80%] text-sm leading-relaxed ${
      role === 'user'
        ? 'bg-teal-600 text-white rounded-tr-md'
        : isError
          ? 'bg-red-50 border border-red-200 text-red-700 rounded-tl-md'
          : 'bg-muted border border-border/50 rounded-tl-md'
    }`}>
      {role === 'user' ? (
        <p className="whitespace-pre-wrap">{content}</p>
      ) : isError ? (
        <div>
          <p className="mb-2">{content}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md bg-red-100 hover:bg-red-200 text-red-700 transition-colors"
            >
              <RotateCcw className="w-3 h-3" /> Retry
            </button>
          )}
        </div>
      ) : (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '')
              const codeString = String(children).replace(/\n$/, '')

              // Inline code (no language tag)
              if (!match) {
                return (
                  <code
                    className="px-1.5 py-0.5 rounded bg-black/10 dark:bg-white/10 text-[0.85em] font-mono"
                    {...props}
                  >
                    {children}
                  </code>
                )
              }

              // Code block with syntax highlighting
              return (
                <div className="relative my-3 rounded-lg overflow-hidden">
                  <div className="flex items-center justify-between px-3 py-1.5 bg-[#282c34] text-xs text-gray-400">
                    <span>{match[1]}</span>
                    <CopyButton text={codeString} />
                  </div>
                  <SyntaxHighlighter
                    style={oneDark}
                    language={match[1]}
                    PreTag="div"
                    customStyle={{ margin: 0, borderRadius: 0, fontSize: '0.8rem' }}
                  >
                    {codeString}
                  </SyntaxHighlighter>
                </div>
              )
            },
            // Style tables from GFM
            table({ children }) {
              return (
                <div className="my-3 overflow-x-auto">
                  <table className="min-w-full text-xs border border-border">
                    {children}
                  </table>
                </div>
              )
            },
            thead({ children }) {
              return <thead className="bg-muted">{children}</thead>
            },
            th({ children }) {
              return (
                <th className="px-3 py-2 text-left border-b border-border font-semibold">
                  {children}
                </th>
              )
            },
            td({ children }) {
              return (
                <td className="px-3 py-2 border-b border-border">{children}</td>
              )
            },
            a({ href, children }) {
              return (
                <a href={href} target="_blank" rel="noopener noreferrer" className="text-teal-600 hover:underline">
                  {children}
                </a>
              )
            },
          }}
        >
          {content}
        </ReactMarkdown>
      )}
    </div>
  )
}

'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import Link from 'next/link'
import {
  Bot, Send, Trash2, ArrowLeft, Settings2, Code2, Search, Hash, Calculator,
  Loader2, ChevronLeft, ChevronRight, Sparkles, Wrench, FileText, Download
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import ReactMarkdown from 'react-markdown'

// ━━━ Types ━━━
interface ChatMessage {
  id: number
  role: 'user' | 'assistant' | 'system'
  content: string
}

interface ToolResult {
  name: string
  result: string
}

const MODEL_OPTIONS = [
  'Qwen 2.5 7B (Best Quality)',
  'Qwen 2.5 3B (Balanced)',
  'Qwen 2.5 0.5B (Fastest)',
  'CodeQwen 7B (Code Specialist)',
]

const QUICK_ACTIONS = [
  { label: 'Code', icon: <Code2 className="w-4 h-4" />, prompt: 'Write a Python function to sort a list using merge sort. Include comments explaining each step.' },
  { label: 'Search', icon: <Search className="w-4 h-4" />, prompt: 'What are the latest breakthroughs in artificial intelligence in 2025?' },
  { label: 'Urdu', icon: <Hash className="w-4 h-4" />, prompt: 'Assalam o Alaikum! Aap kaisay hain? Mujhe aaj ki tareekh aur din bataiye.' },
  { label: 'Math', icon: <Calculator className="w-4 h-4" />, prompt: 'Calculate: (25 * 14) + (376 / 4)' },
]

const DEFAULT_SYSTEM_PROMPT = (
  'You are NexusAI, a helpful, knowledgeable AI assistant created by Osama. '
  + 'You respond fluently in both English and Urdu. '
  + 'Be concise, accurate, and friendly. When writing code, include explanations.'
)

// ━━━ Tool Detection (runs client-side) ━━━
function detectTools(message: string): ToolResult[] {
  const results: ToolResult[] = []

  // Calculator
  const calcMatch = message.match(/(?:calculate|compute|solve|what(?:'s| is))\s+([\d+\-*/().\s]{3,})/i)
  if (calcMatch) {
    const expr = calcMatch[1].trim()
    if (/\d/.test(expr) && /[+\-*/]/.test(expr)) {
      try {
        const allowed = new Set('0123456789+-*/.() ')
        if ([...expr].every(c => allowed.has(c))) {
          const val = eval(expr, { __builtins__: {} } as any)
          results.push({ name: 'Calculator', result: `${expr} = ${val}` })
        }
      } catch { /* skip */ }
    }
  }

  // Date/Time
  if (/(?:what(?:'s| is) the (?:current )?(?:date|time|day)|what time|current (?:date|time)|tariekh|waqt)/i.test(message)) {
    results.push({ name: 'Date/Time', result: new Date().toLocaleString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' }) })
  }

  return results
}

// ━━━ Main Component ━━━
export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [selectedModel, setSelectedModel] = useState(MODEL_OPTIONS[0])
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT)
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  let msgIdCounter = useRef(0)

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Handle pending prompts from quick actions
  useEffect(() => {
    if (pendingPrompt) {
      handleSendMessage(pendingPrompt)
      setPendingPrompt(null)
    }
  }, [pendingPrompt])

  const handleSendMessage = useCallback(async (text?: string) => {
    const messageText = (text || input).trim()
    if (!messageText || isLoading) return

    setInput('')
    msgIdCounter.current++
    const userMsg: ChatMessage = { id: msgIdCounter.current, role: 'user', content: messageText }
    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)

    // Detect tools
    const toolResults = detectTools(messageText)

    // Build API messages
    const apiMessages = [
      { role: 'system' as const, content: systemPrompt },
      ...updatedMessages
        .filter(m => m.role !== 'system')
        .map(m => ({ role: m.role as 'user' | 'assistant', content: m.content })),
    ]

    setIsLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: apiMessages,
          model: selectedModel,
          temperature,
          max_tokens: maxTokens,
          tool_results: toolResults.length > 0 ? toolResults : undefined,
        }),
      })

      const data = await res.json()

      msgIdCounter.current++
      const assistantMsg: ChatMessage = {
        id: msgIdCounter.current,
        role: 'assistant',
        content: data.error || data.content || 'No response received.',
      }
      setMessages(prev => [...prev, assistantMsg])
    } catch {
      msgIdCounter.current++
      setMessages(prev => [
        ...prev,
        { id: msgIdCounter.current, role: 'assistant', content: 'Network error. Please check your connection and try again.' },
      ])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }, [input, isLoading, messages, selectedModel, temperature, maxTokens, systemPrompt])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      {/* ━━━ Top Bar ━━━ */}
      <header className="flex items-center justify-between h-14 px-4 border-b border-border/50 bg-background/80 backdrop-blur-xl shrink-0">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" asChild className="gap-1.5 text-muted-foreground hover:text-foreground">
            <Link href="/"><ArrowLeft className="w-4 h-4" /> Home</Link>
          </Button>
          <div className="hidden sm:block w-px h-6 bg-border" />
          <div className="hidden sm:flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold tracking-tight">Nexus<span className="text-teal-600">AI</span> Chat</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost" size="sm"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="gap-1.5 text-muted-foreground"
          >
            <Settings2 className="w-4 h-4" />
            <span className="hidden sm:inline">Settings</span>
            {sidebarOpen ? <ChevronRight className="w-3 h-3" /> : <ChevronLeft className="w-3 h-3" />}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setMessages([])} className="gap-1.5 text-muted-foreground hover:text-red-500">
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* ━━━ Sidebar ━━━ */}
        {sidebarOpen && (
          <aside className="w-72 shrink-0 border-r border-border/50 bg-muted/30 overflow-y-auto hidden md:block">
            <div className="p-4 space-y-5">
              {/* Model Selection */}
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                  AI Model
                </label>
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                  className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500"
                >
                  {MODEL_OPTIONS.map(m => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>

              {/* System Prompt */}
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                  System Prompt
                </label>
                <textarea
                  value={systemPrompt}
                  onChange={e => setSystemPrompt(e.target.value)}
                  rows={4}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 resize-none"
                />
              </div>

              {/* Temperature */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Temperature
                  </label>
                  <span className="text-xs text-teal-600 font-mono">{temperature}</span>
                </div>
                <input
                  type="range" min="0.1" max="1.5" step="0.1" value={temperature}
                  onChange={e => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-teal-600"
                />
              </div>

              {/* Max Tokens */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Max Tokens
                  </label>
                  <span className="text-xs text-teal-600 font-mono">{maxTokens}</span>
                </div>
                <input
                  type="range" min="128" max="4096" step="128" value={maxTokens}
                  onChange={e => setMaxTokens(parseInt(e.target.value))}
                  className="w-full accent-teal-600"
                />
              </div>

              <hr className="border-border/50" />

              {/* Quick Actions */}
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                  Quick Actions
                </label>
                <div className="grid grid-cols-2 gap-2">
                  {QUICK_ACTIONS.map(qa => (
                    <button
                      key={qa.label}
                      onClick={() => setPendingPrompt(qa.prompt)}
                      disabled={isLoading}
                      className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md border border-border bg-background hover:bg-muted transition-colors disabled:opacity-50"
                    >
                      {qa.icon} {qa.label}
                    </button>
                  ))}
                </div>
              </div>

              <hr className="border-border/50" />

              {/* Tools Info */}
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <Wrench className="w-3.5 h-3.5 text-muted-foreground" />
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Agent Tools
                  </label>
                </div>
                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2"><Calculator className="w-3 h-3" /> Calculator (auto)</div>
                  <div className="flex items-center gap-2"><Code2 className="w-3 h-3" /> Code Executor</div>
                  <div className="flex items-center gap-2"><Sparkles className="w-3 h-3" /> Date/Time (auto)</div>
                  <div className="flex items-center gap-2"><Search className="w-3 h-3" /> Web Search (placeholder)</div>
                </div>
              </div>

              <hr className="border-border/50" />

              {/* Download links */}
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Resources
                  </label>
                </div>
                <div className="space-y-1.5">
                  <a href="/docs/NexusAI_Architecture.pdf" download className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
                    <Download className="w-3 h-3" /> Architecture PDF
                  </a>
                  <a href="/docs/NexusAI_Business_Plan.docx" download className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
                    <Download className="w-3 h-3" /> Business Plan
                  </a>
                </div>
              </div>
            </div>
          </aside>
        )}

        {/* ━━━ Chat Area ━━━ */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Mobile: model selector + quick actions bar */}
          <div className="md:hidden flex items-center gap-2 px-4 py-2 border-b border-border/50 overflow-x-auto">
            <select
              value={selectedModel}
              onChange={e => setSelectedModel(e.target.value)}
              className="h-8 rounded-md border border-input bg-background px-2 text-xs focus:outline-none"
            >
              {MODEL_OPTIONS.map(m => (
                <option key={m} value={m}>{m.split('(')[0].trim()}</option>
              ))}
            </select>
            {QUICK_ACTIONS.map(qa => (
              <button
                key={qa.label}
                onClick={() => setPendingPrompt(qa.prompt)}
                disabled={isLoading}
                className="shrink-0 px-2.5 py-1 text-xs font-medium rounded-full border border-border bg-background hover:bg-muted transition-colors disabled:opacity-50"
              >
                {qa.icon} {qa.label}
              </button>
            ))}
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-6">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center mb-4">
                  <Bot className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-xl font-bold">NexusAI Chat</h2>
                <p className="text-sm text-muted-foreground mt-2 max-w-sm">
                  Free multi-model AI chat. Supports English & Urdu. Try a quick action or type your message below.
                </p>
                <div className="flex flex-wrap justify-center gap-2 mt-6">
                  {QUICK_ACTIONS.map(qa => (
                    <button
                      key={qa.label}
                      onClick={() => setPendingPrompt(qa.prompt)}
                      className="flex items-center gap-1.5 px-4 py-2 text-sm rounded-full border border-border bg-background hover:bg-muted transition-colors"
                    >
                      {qa.icon} {qa.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map(msg => (
              <div
                key={msg.id}
                className={`max-w-3xl mx-auto mb-4 flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                  msg.role === 'user'
                    ? 'bg-teal-100 text-teal-700'
                    : 'bg-gradient-to-br from-teal-500 to-cyan-600 text-white'
                }`}>
                  {msg.role === 'user' ? <span className="text-sm font-bold">O</span> : <Bot className="w-4 h-4" />}
                </div>
                <div className={`rounded-2xl px-4 py-3 max-w-[80%] text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-teal-600 text-white rounded-tr-md'
                    : 'bg-muted border border-border/50 rounded-tl-md'
                }`}>
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="max-w-3xl mx-auto mb-4 flex gap-3">
                <div className="shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center text-white">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="bg-muted border border-border/50 rounded-2xl rounded-tl-md px-4 py-3">
                  <Loader2 className="w-4 h-4 animate-spin text-teal-600" />
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div className="shrink-0 border-t border-border/50 bg-background/80 backdrop-blur-xl p-4">
            <div className="max-w-3xl mx-auto flex items-end gap-3">
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your message... (English or Urdu)"
                rows={1}
                className="flex-1 resize-none rounded-xl border border-input bg-background px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 min-h-[44px] max-h-[120px]"
                style={{ height: 'auto' }}
                onInput={e => { const t = e.target as HTMLTextAreaElement; t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, 120) + 'px' }}
              />
              <Button
                onClick={() => handleSendMessage()}
                disabled={!input.trim() || isLoading}
                className="h-11 w-11 rounded-xl bg-teal-600 hover:bg-teal-700 text-white p-0 shrink-0"
              >
                {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </Button>
            </div>
            <p className="text-center text-[11px] text-muted-foreground mt-2">
              Powered by Qwen 2.5 via HuggingFace Inference API | 100% Free
            </p>
          </div>
        </main>
      </div>
    </div>
  )
}

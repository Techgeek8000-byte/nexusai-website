'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import Link from 'next/link'
import {
  Bot, Send, Trash2, ArrowLeft, Settings2, Code2, Search, Hash, Calculator,
  Loader2, ChevronLeft, ChevronRight, Sparkles, Wrench, FileText, Download,
  Plus, MessageSquare, Copy, Check, PanelLeftClose, PanelLeft
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import ReactMarkdown from 'react-markdown'

// ━━━ Types ━━━
interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
}

interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  model: string
  createdAt: number
  updatedAt: number
}

const MODEL_OPTIONS = [
  'Qwen 2.5 7B (Best Quality)',
  'Qwen 2.5 3B (Balanced)',
  'Qwen 2.5 0.5B (Fastest)',
  'CodeQwen 7B (Code Specialist)',
]

const QUICK_ACTIONS = [
  { label: 'Code', icon: <Code2 className="w-4 h-4" />, prompt: 'Write a Python function to sort a list using merge sort. Include comments.' },
  { label: 'Search', icon: <Search className="w-4 h-4" />, prompt: 'What are the latest breakthroughs in artificial intelligence in 2025?' },
  { label: 'Urdu', icon: <Hash className="w-4 h-4" />, prompt: 'Assalam o Alaikum! Aap kaisay hain? Mujhe aaj ki tareekh aur din bataiye.' },
  { label: 'Math', icon: <Calculator className="w-4 h-4" />, prompt: 'Calculate: (25 * 14) + (376 / 4)' },
]

const DEFAULT_SYSTEM_PROMPT = (
  'You are NexusAI, a helpful, knowledgeable AI assistant created by Osama. '
  + 'You respond fluently in both English and Urdu. '
  + 'Be concise, accurate, and friendly. When writing code, include explanations.'
)

const LS_CONVERSATIONS = 'nexusai-convs'
const LS_ACTIVE = 'nexusai-active'
const LS_SETTINGS = 'nexusai-settings'

// ━━━ Tool Detection ━━━
function detectTools(message: string): { name: string; result: string }[] {
  const results: { name: string; result: string }[] = []
  const calcMatch = message.match(/(?:calculate|compute|solve|what(?:'s| is))\s+([\d+\-*/().\s]{3,})/i)
  if (calcMatch) {
    const expr = calcMatch[1].trim()
    if (/\d/.test(expr) && /[+\-*/]/.test(expr)) {
      try {
        const allowed = new Set('0123456789+-*/.() ')
        if ([...expr].every(c => allowed.has(c))) {
          results.push({ name: 'Calculator', result: `${expr} = ${eval(expr, { __builtins__: {} } as any)}` })
        }
      } catch { /* skip */ }
    }
  }
  if (/(?:what(?:'s| is) the (?:current )?(?:date|time|day)|what time|current (?:date|time))/i.test(message)) {
    results.push({ name: 'Date/Time', result: new Date().toLocaleString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' }) })
  }
  return results
}

// ━━━ Copy Button Component ━━━
function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 2000) }}
      className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-white/10"
      title="Copy"
    >
      {copied ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
    </button>
  )
}

// ━━━ Code Block Component ━━━
function CodeBlock({ children, className }: { children: string; className?: string }) {
  const code = typeof children === 'string' ? children : String(children)
  const lang = className?.replace('language-', '') || ''
  const [copied, setCopied] = useState(false)
  return (
    <div className="relative group rounded-lg overflow-hidden my-2">
      <div className="flex items-center justify-between px-4 py-1.5 bg-zinc-800 text-zinc-400 text-xs">
        <span>{lang || 'code'}</span>
        <button
          onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 2000) }}
          className="flex items-center gap-1 hover:text-white transition-colors"
        >
          {copied ? <><Check className="w-3 h-3" /> Copied</> : <><Copy className="w-3 h-3" /> Copy</>}
        </button>
      </div>
      <pre className="bg-zinc-900 p-4 overflow-x-auto text-sm leading-relaxed">
        <code className={className}>{code}</code>
      </pre>
    </div>
  )
}

// ━━━ Main Component ━━━
export default function ChatPage() {
  // Conversations
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamText, setStreamText] = useState('')
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null)
  const [inputVal, setInputVal] = useState('')

  // UI
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [showSettings, setShowSettings] = useState(false)
  const [copiedMsgId, setCopiedMsgId] = useState<string | null>(null)

  // Settings
  const [settings, setSettings] = useState({
    model: MODEL_OPTIONS[0],
    temperature: 0.7,
    maxTokens: 2048,
    systemPrompt: DEFAULT_SYSTEM_PROMPT,
  })

  const chatEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const activeConv = conversations.find(c => c.id === activeId)

  // ── Load from localStorage ──
  useEffect(() => {
    try {
      const saved = localStorage.getItem(LS_CONVERSATIONS)
      if (saved) setConversations(JSON.parse(saved))
    } catch {}
    try {
      const savedId = localStorage.getItem(LS_ACTIVE)
      if (savedId) setActiveId(savedId)
    } catch {}
    try {
      const savedSettings = localStorage.getItem(LS_SETTINGS)
      if (savedSettings) setSettings(JSON.parse(savedSettings))
    } catch {}
  }, [])

  // ── Persist to localStorage ──
  useEffect(() => {
    try { localStorage.setItem(LS_CONVERSATIONS, JSON.stringify(conversations)) } catch {}
  }, [conversations])
  useEffect(() => {
    try { localStorage.setItem(LS_ACTIVE, activeId || '') } catch {}
  }, [activeId])
  useEffect(() => {
    try { localStorage.setItem(LS_SETTINGS, JSON.stringify(settings)) } catch {}
  }, [settings])

  // ── Auto-scroll ──
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeConv?.messages, streamText])

  // ── Handle pending quick action prompts ──
  useEffect(() => {
    if (pendingPrompt) {
      sendMessage(pendingPrompt)
      setPendingPrompt(null)
    }
  }, [pendingPrompt])

  // ── Create new chat ──
  const createNewChat = useCallback(() => {
    const newConv: Conversation = {
      id: crypto.randomUUID(), title: 'New Chat', messages: [],
      model: settings.model, createdAt: Date.now(), updatedAt: Date.now(),
    }
    setConversations(prev => [newConv, ...prev])
    setActiveId(newConv.id)
    return newConv.id
  }, [settings.model])

  // ── Delete chat ──
  const deleteChat = useCallback((id: string) => {
    setConversations(prev => prev.filter(c => c.id !== id))
    if (activeId === id) {
      setActiveId(prev => {
        const remaining = conversations.filter(c => c.id !== id)
        return remaining.length > 0 ? remaining[0].id : null
      })
    }
  }, [activeId, conversations])

  // ── Export chat as Markdown ──
  const exportChat = useCallback(() => {
    if (!activeConv) return
    const lines = activeConv.messages.map(m =>
      m.role === 'user' ? `## You\n${m.content}` : `## NexusAI\n${m.content}`
    )
    const md = `# NexusAI Chat Export\n\nModel: ${activeConv.model}\nDate: ${new Date(activeConv.updatedAt).toLocaleString()}\n\n---\n\n${lines.join('\n\n---\n\n')}`
    const blob = new Blob([md], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `nexusai-chat-${Date.now()}.md`; a.click()
    URL.revokeObjectURL(url)
  }, [activeConv])

  // ── Send message with streaming ──
  const sendMessage = useCallback(async (text?: string) => {
    const messageText = (text || inputVal).trim()
    if (!messageText || isStreaming) return
    setInputVal('')

    // Ensure active conversation
    let targetId = activeId
    if (!targetId) {
      targetId = createNewChat()
    }

    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: messageText }
    const conv = conversations.find(c => c.id === targetId)
    const existingMsgs = conv?.messages || []
    const updatedMsgs = [...existingMsgs, userMsg]

    // Update title on first message
    const newTitle = existingMsgs.length === 0
      ? (messageText.length > 50 ? messageText.slice(0, 50) + '...' : messageText)
      : (conv?.title || 'New Chat')

    setConversations(prev => prev.map(c =>
      c.id === targetId ? { ...c, messages: updatedMsgs, updatedAt: Date.now(), title: newTitle } : c
    ))

    // Tool detection
    const toolResults = detectTools(messageText)

    // Build API messages
    const apiMessages = [
      { role: 'system' as const, content: settings.systemPrompt },
      ...existingMsgs.map(m => ({ role: m.role as 'user' | 'assistant', content: m.content })),
      { role: 'user' as const, content: messageText },
    ]

    setIsStreaming(true)
    setStreamText('')

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: apiMessages, model: settings.model,
          temperature: settings.temperature, max_tokens: settings.maxTokens,
          tool_results: toolResults.length > 0 ? toolResults : undefined,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: 'Request failed' }))
        throw new Error(err.error || `HTTP ${res.status}`)
      }

      // Parse SSE stream
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let fullContent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')
        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed.startsWith('data: ')) continue
          const data = trimmed.slice(6)
          if (data === '[DONE]') continue
          try {
            const parsed = JSON.parse(data)
            const delta = parsed.choices?.[0]?.delta?.content
            if (delta) { fullContent += delta; setStreamText(fullContent) }
          } catch { /* skip malformed chunks */ }
        }
      }

      // Finalize
      const assistantMsg: ChatMessage = { id: crypto.randomUUID(), role: 'assistant', content: fullContent || 'No response received.' }
      const finalTargetId = targetId
      setConversations(prev => prev.map(c =>
        c.id === finalTargetId ? { ...c, messages: [...updatedMsgs, assistantMsg], updatedAt: Date.now() } : c
      ))
    } catch (error) {
      const errMsg: ChatMessage = { id: crypto.randomUUID(), role: 'assistant', content: `Error: ${error instanceof Error ? error.message : String(error)}` }
      const finalTargetId = targetId
      setConversations(prev => prev.map(c =>
        c.id === finalTargetId ? { ...c, messages: [...updatedMsgs, errMsg], updatedAt: Date.now() } : c
      ))
    } finally {
      setIsStreaming(false)
      setStreamText('')
      inputRef.current?.focus()
    }
  }, [inputVal, isStreaming, activeId, conversations, settings, createNewChat])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  const copyMessage = (msg: ChatMessage) => {
    navigator.clipboard.writeText(msg.content)
    setCopiedMsgId(msg.id)
    setTimeout(() => setCopiedMsgId(null), 2000)
  }

  const allMessages = activeConv ? [...activeConv.messages] : []
  const displayMessages = isStreaming
    ? [...allMessages, { id: '__streaming__', role: 'assistant' as const, content: streamText }]
    : allMessages

  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      {/* ━━━ Top Bar ━━━ */}
      <header className="flex items-center justify-between h-14 px-4 border-b border-border/50 bg-background/80 backdrop-blur-xl shrink-0 z-10">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => setSidebarOpen(!sidebarOpen)} className="gap-1 text-muted-foreground">
            {sidebarOpen ? <PanelLeftClose className="w-4 h-4" /> : <PanelLeft className="w-4 h-4" />}
          </Button>
          <div className="hidden sm:block w-px h-6 bg-border" />
          <Button variant="ghost" size="sm" asChild className="hidden sm:flex gap-1.5 text-muted-foreground hover:text-foreground">
            <Link href="/"><ArrowLeft className="w-4 h-4" /> Home</Link>
          </Button>
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold tracking-tight hidden sm:inline">Nexus<span className="text-teal-600">AI</span></span>
          </div>
        </div>
        <div className="flex items-center gap-1">
          {activeConv && activeConv.messages.length > 0 && (
            <Button variant="ghost" size="sm" onClick={exportChat} className="gap-1 text-muted-foreground" title="Export chat">
              <Download className="w-4 h-4" /><span className="hidden sm:inline text-xs">Export</span>
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={() => { if (activeId) deleteChat(activeId) }} className="gap-1 text-muted-foreground hover:text-red-500" title="Delete chat">
            <Trash2 className="w-4 h-4" />
          </Button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* ━━━ Left Sidebar — Conversations ━━━ */}
        {sidebarOpen && (
          <aside className="w-64 shrink-0 border-r border-border/50 bg-muted/30 flex flex-col overflow-hidden">
            <div className="p-3">
              <Button onClick={createNewChat} className="w-full gap-2 bg-teal-600 hover:bg-teal-700 text-white" size="sm">
                <Plus className="w-4 h-4" /> New Chat
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto px-2 pb-2">
              {conversations.length === 0 && (
                <p className="text-xs text-muted-foreground text-center py-8">No conversations yet</p>
              )}
              {conversations.map(conv => (
                <button
                  key={conv.id}
                  onClick={() => { setActiveId(conv.id); setShowSettings(false) }}
                  className={`w-full text-left px-3 py-2.5 rounded-lg mb-0.5 text-sm transition-colors group flex items-start gap-2 ${
                    conv.id === activeId ? 'bg-teal-50 text-teal-900 border border-teal-200' : 'hover:bg-muted text-foreground'
                  }`}
                >
                  <MessageSquare className="w-4 h-4 mt-0.5 shrink-0 opacity-50" />
                  <div className="flex-1 min-w-0">
                    <div className="truncate font-medium">{conv.title}</div>
                    <div className="text-[10px] opacity-50 mt-0.5">
                      {conv.messages.length} msgs · {new Date(conv.updatedAt).toLocaleDateString()}
                    </div>
                  </div>
                </button>
              ))}
            </div>
            <div className="p-3 border-t border-border/50">
              <Button
                variant={showSettings ? 'secondary' : 'ghost'}
                size="sm"
                className="w-full gap-2 justify-start"
                onClick={() => setShowSettings(!showSettings)}
              >
                <Settings2 className="w-4 h-4" /> Settings
              </Button>
            </div>
          </aside>
        )}

        {/* ━━━ Right Panel — Settings (shown when toggled) ━━━ */}
        {showSettings && sidebarOpen && (
          <aside className="w-72 shrink-0 border-r border-border/50 bg-muted/20 overflow-y-auto">
            <div className="p-4 space-y-5">
              <h3 className="text-sm font-semibold">Chat Settings</h3>
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1.5 block">AI Model</label>
                <select value={settings.model} onChange={e => setSettings(s => ({ ...s, model: e.target.value }))}
                  className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500">
                  {MODEL_OPTIONS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1.5 block">System Prompt</label>
                <textarea value={settings.systemPrompt} onChange={e => setSettings(s => ({ ...s, systemPrompt: e.target.value }))}
                  rows={4} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 resize-none" />
              </div>
              <div>
                <div className="flex justify-between mb-1.5">
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Temperature</label>
                  <span className="text-xs text-teal-600 font-mono">{settings.temperature}</span>
                </div>
                <input type="range" min="0.1" max="1.5" step="0.1" value={settings.temperature}
                  onChange={e => setSettings(s => ({ ...s, temperature: parseFloat(e.target.value) }))} className="w-full accent-teal-600" />
              </div>
              <div>
                <div className="flex justify-between mb-1.5">
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Max Tokens</label>
                  <span className="text-xs text-teal-600 font-mono">{settings.maxTokens}</span>
                </div>
                <input type="range" min="128" max="4096" step="128" value={settings.maxTokens}
                  onChange={e => setSettings(s => ({ ...s, maxTokens: parseInt(e.target.value) }))} className="w-full accent-teal-600" />
              </div>
              <hr className="border-border/50" />
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">Quick Actions</label>
                <div className="grid grid-cols-2 gap-2">
                  {QUICK_ACTIONS.map(qa => (
                    <button key={qa.label} onClick={() => setPendingPrompt(qa.prompt)} disabled={isStreaming}
                      className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md border border-border bg-background hover:bg-muted transition-colors disabled:opacity-50">
                      {qa.icon} {qa.label}
                    </button>
                  ))}
                </div>
              </div>
              <hr className="border-border/50" />
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <Wrench className="w-3.5 h-3.5 text-muted-foreground" />
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Tools (auto)</label>
                </div>
                <div className="space-y-1 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2"><Calculator className="w-3 h-3" /> Calculator</div>
                  <div className="flex items-center gap-2"><Sparkles className="w-3 h-3" /> Date / Time</div>
                </div>
              </div>
              <hr className="border-border/50" />
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Docs</label>
                </div>
                <div className="space-y-1.5">
                  <a href="/docs/NexusAI_Architecture.pdf" download className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
                    <Download className="w-3 h-3" /> Architecture PDF</a>
                  <a href="/docs/NexusAI_Business_Plan.docx" download className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground">
                    <Download className="w-3 h-3" /> Business Plan</a>
                </div>
              </div>
            </div>
          </aside>
        )}

        {/* ━━━ Chat Area ━━━ */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Mobile model + quick actions */}
          <div className="md:hidden flex items-center gap-2 px-3 py-2 border-b border-border/50 overflow-x-auto">
            <select value={settings.model} onChange={e => setSettings(s => ({ ...s, model: e.target.value }))}
              className="h-7 rounded border border-input bg-background px-2 text-[11px] focus:outline-none shrink-0">
              {MODEL_OPTIONS.map(m => <option key={m} value={m}>{m.split('(')[0].trim()}</option>)}
            </select>
            {QUICK_ACTIONS.map(qa => (
              <button key={qa.label} onClick={() => setPendingPrompt(qa.prompt)} disabled={isStreaming}
                className="shrink-0 px-2 py-1 text-[11px] rounded-full border border-border bg-background hover:bg-muted disabled:opacity-50">
                {qa.icon} {qa.label}
              </button>
            ))}
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-6">
            {!activeConv || activeConv.messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center mb-4">
                  <Bot className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-xl font-bold">NexusAI Chat</h2>
                <p className="text-sm text-muted-foreground mt-2 max-w-sm">
                  Free multi-model AI chat. English & Urdu. Start a conversation or try a quick action.
                </p>
                <div className="flex flex-wrap justify-center gap-2 mt-6">
                  {QUICK_ACTIONS.map(qa => (
                    <button key={qa.label} onClick={() => setPendingPrompt(qa.prompt)}
                      className="flex items-center gap-1.5 px-4 py-2 text-sm rounded-full border border-border bg-background hover:bg-muted">
                      {qa.icon} {qa.label}
                    </button>
                  ))}
                </div>
                <p className="text-[11px] text-muted-foreground/50 mt-8">
                  Messages are saved locally in your browser.
                </p>
              </div>
            ) : (
              displayMessages.map(msg => (
                <div key={msg.id} className={`max-w-3xl mx-auto mb-4 flex gap-3 group ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  <div className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                    msg.role === 'user' ? 'bg-teal-100 text-teal-700' : 'bg-gradient-to-br from-teal-500 to-cyan-600 text-white'
                  }`}
                  >
                    {msg.role === 'user' ? <span className="text-sm font-bold">O</span> : <Bot className="w-4 h-4" />}
                  </div>
                  <div className="relative max-w-[80%]">
                    <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                      msg.role === 'user'
                        ? 'bg-teal-600 text-white rounded-tr-md'
                        : 'bg-muted border border-border/50 rounded-tl-md'
                    }`}>
                      <ReactMarkdown components={{
                        code({ children, className, ...props }) {
                          const isInline = !className && typeof children === 'string' && !children.includes('\n')
                          if (isInline) return <code className="bg-black/10 px-1.5 py-0.5 rounded text-[13px]" {...props}>{children}</code>
                          return <CodeBlock className={className}>{String(children)}</CodeBlock>
                        },
                        p({ children }) { return <p className="mb-2 last:mb-0">{children}</p> },
                        ul({ children }) { return <ul className="list-disc pl-5 mb-2 space-y-1">{children}</ul> },
                        ol({ children }) { return <ol className="list-decimal pl-5 mb-2 space-y-1">{children}</ol> },
                        h1({ children }) { return <h1 className="text-lg font-bold mt-4 mb-2">{children}</h1> },
                        h2({ children }) { return <h2 className="text-base font-bold mt-3 mb-2">{children}</h2> },
                        h3({ children }) { return <h3 className="text-sm font-bold mt-2 mb-1">{children}</h3> },
                      }}>{msg.content}</ReactMarkdown>
                      {msg.id === '__streaming__' && (
                        <span className="inline-block w-2 h-4 bg-teal-500 animate-pulse ml-0.5 align-middle" />
                      )}
                    </div>
                    {msg.id !== '__streaming__' && (
                      <div className={`flex items-center gap-1 mt-1 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                        <button onClick={() => copyMessage(msg)} className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-muted text-muted-foreground">
                          {copiedMsgId === msg.id ? <Check className="w-3 h-3 text-green-500" /> : <Copy className="w-3 h-3" />}
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div className="shrink-0 border-t border-border/50 bg-background/80 backdrop-blur-xl p-4">
            <div className="max-w-3xl mx-auto flex items-end gap-3">
              <textarea
                ref={inputRef}
                value={inputVal}
                onChange={e => setInputVal(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your message... (English or Urdu)"
                rows={1}
                className="flex-1 resize-none rounded-xl border border-input bg-background px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 min-h-[44px] max-h-[120px]"
                onInput={e => { const t = e.target as HTMLTextAreaElement; t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, 120) + 'px' }}
              />
              <Button onClick={() => sendMessage()} disabled={!inputVal.trim() || isStreaming}
                className="h-11 w-11 rounded-xl bg-teal-600 hover:bg-teal-700 text-white p-0 shrink-0">
                {isStreaming ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </Button>
            </div>
            <p className="text-center text-[11px] text-muted-foreground mt-2">
              Qwen 2.5 via HuggingFace · Streaming · Saved locally · 100% Free
            </p>
          </div>
        </main>
      </div>
    </div>
  )
}

'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import Link from 'next/link'
import {
  Bot, Send, Trash2, ArrowLeft, Settings2, Code2, Search, Hash, Calculator,
  Loader2, ChevronLeft, ChevronRight, Sparkles, Wrench, FileText, Download,
  Plus, MessageSquare, Clock, Upload, FileDown, RefreshCw, Terminal, Globe
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ChatMessageBubble } from '@/components/chat-message'
import { detectToolsClient } from '@/lib/tools'
import {
  type SavedConversation, loadConversations, saveConversation,
  deleteConversation, getActiveConversationId, setActiveConversationId,
  generateId, generateTitle
} from '@/lib/storage'

// ━━━ Types ━━━
interface ChatMessage {
  id: number
  role: 'user' | 'assistant' | 'system'
  content: string
  isError?: boolean
}

interface LastRequest {
  text: string
  messages: ChatMessage[]
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
  + 'Be concise, accurate, and friendly. When writing code, include explanations. '
  + 'When the user asks you to execute or run code, use the execute_code tool. '
  + 'When the user asks about current information, use the web_search tool.'
)

// ━━━ Main Component ━━━
export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(true)
  const [conversations, setConversations] = useState<SavedConversation[]>([])
  const [activeConvId, setActiveConvId] = useState<string | null>(null)
  const [selectedModel, setSelectedModel] = useState(MODEL_OPTIONS[0])
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(2048)
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT)
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null)
  const [attachedFile, setAttachedFile] = useState<{ name: string; content: string } | null>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const msgIdCounter = useRef(0)
  const lastRequestRef = useRef<LastRequest | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // ── Load conversations on mount ──
  useEffect(() => {
    const saved = loadConversations()
    setConversations(saved)
    const activeId = getActiveConversationId()
    if (activeId) {
      const conv = saved.find(c => c.id === activeId)
      if (conv) {
        setActiveConvId(activeId)
        setMessages(conv.messages.map((m, i) => ({ id: i + 1, role: m.role as 'user' | 'assistant', content: m.content })))
        if (conv.model) setSelectedModel(conv.model)
        msgIdCounter.current = conv.messages.length
      }
    }
  }, [])

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Handle pending prompts
  useEffect(() => {
    if (pendingPrompt) {
      handleSendMessage(pendingPrompt)
      setPendingPrompt(null)
    }
  }, [pendingPrompt])

  // ── Persist ──
  const persistMessages = useCallback((msgs: ChatMessage[], convId: string | null, model: string) => {
    const chatMsgs = msgs.filter(m => m.role !== 'system' && !m.isError).map(m => ({ role: m.role as 'user' | 'assistant', content: m.content }))
    if (chatMsgs.length === 0) return
    const id = convId || generateId()
    const firstUserMsg = chatMsgs.find(m => m.role === 'user')?.content || 'New Chat'
    const title = generateTitle(firstUserMsg)
    const now = Date.now()
    const existing = loadConversations().find(c => c.id === id)
    const conv: SavedConversation = {
      id,
      title: existing?.title || title,
      messages: chatMsgs,
      model,
      createdAt: existing?.createdAt || now,
      updatedAt: now,
    }
    saveConversation(conv)
    setConversations(loadConversations())
    setActiveConvId(id)
    setActiveConversationId(id)
    return id
  }, [])

  const startNewChat = useCallback(() => {
    setMessages([])
    setActiveConvId(null)
    setActiveConversationId(null)
    msgIdCounter.current = 0
    setAttachedFile(null)
    inputRef.current?.focus()
  }, [])

  const loadConversation = useCallback((conv: SavedConversation) => {
    setMessages(conv.messages.map((m, i) => ({ id: i + 1, role: m.role as 'user' | 'assistant', content: m.content })))
    setActiveConvId(conv.id)
    setActiveConversationId(conv.id)
    if (conv.model) setSelectedModel(conv.model)
    msgIdCounter.current = conv.messages.length
    setHistoryOpen(false)
    setAttachedFile(null)
  }, [])

  const handleDeleteConversation = useCallback((id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    deleteConversation(id)
    setConversations(loadConversations())
    if (activeConvId === id) startNewChat()
  }, [activeConvId, startNewChat])

  // ── File Upload ──
  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    if (file.size > 100_000) {
      alert('File too large. Maximum 100KB.')
      return
    }
    const reader = new FileReader()
    reader.onload = () => {
      setAttachedFile({ name: file.name, content: reader.result as string })
    }
    reader.readAsText(file)
    e.target.value = ''
  }, [])

  // ── Export Conversation ──
  const handleExport = useCallback(() => {
    const chatMsgs = messages.filter(m => m.role !== 'system')
    let md = `# NexusAI Chat Export\n\n`
    chatMsgs.forEach(m => {
      md += `### ${m.role === 'user' ? 'You' : 'NexusAI'}\n\n${m.content}\n\n---\n\n`
    })
    const blob = new Blob([md], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `nexusai-chat-${new Date().toISOString().slice(0, 10)}.md`
    a.click()
    URL.revokeObjectURL(url)
  }, [messages])

  // ── Send Message (with streaming) ──
  const handleSendMessage = useCallback(async (text?: string) => {
    const messageText = (text || input).trim()
    if (!messageText || isLoading) return

    setInput('')
    msgIdCounter.current++
    const userMsg: ChatMessage = { id: msgIdCounter.current, role: 'user', content: messageText }
    const updatedMessages = [...messages, userMsg]
    setMessages(updatedMessages)

    // Build full message with file content
    let fullContent = messageText
    if (attachedFile) {
      fullContent = `[User uploaded file: ${attachedFile.name}]\n\n${attachedFile.content}\n\n---\n\n${messageText}`
      setAttachedFile(null)
    }

    // Detect quick tools
    const toolResults = detectToolsClient(messageText)

    // Build API messages
    const apiMessages = [
      { role: 'system' as const, content: systemPrompt },
      ...updatedMessages.filter(m => m.role !== 'system').map(m => ({ role: m.role as 'user' | 'assistant', content: m.content })),
    ]
    // Replace last user message with full content if file was attached
    if (attachedFile || (fullContent !== messageText)) {
      const lastUserIdx = apiMessages.length - 1
      if (apiMessages[lastUserIdx].role === 'user') {
        apiMessages[lastUserIdx].content = fullContent
      }
    }

    // Save for retry
    lastRequestRef.current = { text: messageText, messages: updatedMessages }

    setIsLoading(true)
    abortRef.current = new AbortController()

    // Create placeholder assistant message for streaming
    msgIdCounter.current++
    const assistantId = msgIdCounter.current
    setMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '' }])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: apiMessages,
          model: selectedModel,
          temperature,
          max_tokens: maxTokens,
          stream: true,
          tool_results: toolResults.length > 0 ? toolResults : undefined,
        }),
        signal: abortRef.current.signal,
      })

      // Check if response is SSE (streaming) or JSON
      const contentType = res.headers.get('content-type') || ''

      if (contentType.includes('text/event-stream')) {
        // ── Read SSE stream ──
        const reader = res.body?.getReader()
        if (!reader) throw new Error('No response body')

        const decoder = new TextDecoder()
        let fullContent = ''
        let buffer = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            const data = line.slice(6).trim()
            if (data === '[DONE]') continue
            try {
              const parsed = JSON.parse(data)
              if (parsed.error) {
                setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: parsed.error, isError: true } : m))
                fullContent = parsed.error
                break
              }
              if (parsed.model) continue // skip model event
              const delta = parsed.choices?.[0]?.delta?.content
              if (delta) {
                fullContent += delta
                setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: fullContent } : m))
              }
            } catch { /* skip */ }
          }
        }
      } else {
        // ── Non-streaming JSON fallback ──
        const data = await res.json()
        const reply = data.error || data.content || 'No response received.'
        const isErr = !!data.error
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: reply, isError: isErr } : m))
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return
      setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: 'Network error. Check your connection and try again.', isError: true } : m))
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
      // Persist after response
      setMessages(currentMsgs => {
        persistMessages(currentMsgs, activeConvId, selectedModel)
        return currentMsgs
      })
    }
  }, [input, isLoading, messages, selectedModel, temperature, maxTokens, systemPrompt, activeConvId, persistMessages, attachedFile])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const stopGeneration = () => {
    abortRef.current?.abort()
    setIsLoading(false)
  }

  const handleRetry = useCallback(async () => {
    const last = lastRequestRef.current
    if (!last) return
    setMessages(prev => prev.slice(0, -1))
    // Re-send with the same text
    const messageText = last.text
    if (!messageText) return
    setInput('')
    msgIdCounter.current++
    const userMsg: ChatMessage = { id: msgIdCounter.current, role: 'user', content: messageText }
    const updatedMessages = [...messages.slice(0, -1), userMsg]
    setMessages(updatedMessages)
    const toolResults = detectToolsClient(messageText)
    const apiMessages = [
      { role: 'system' as const, content: systemPrompt },
      ...updatedMessages.filter(m => m.role !== 'system').map(m => ({ role: m.role as 'user' | 'assistant', content: m.content })),
    ]
    lastRequestRef.current = { text: messageText, messages: updatedMessages }
    setIsLoading(true)
    abortRef.current = new AbortController()
    msgIdCounter.current++
    const assistantId = msgIdCounter.current
    setMessages(prev => [...prev, { id: assistantId, role: 'assistant', content: '' }])
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: apiMessages, model: selectedModel, temperature, max_tokens: maxTokens, stream: true, tool_results: toolResults.length > 0 ? toolResults : undefined }),
        signal: abortRef.current.signal,
      })
      const contentType = res.headers.get('content-type') || ''
      if (contentType.includes('text/event-stream')) {
        const reader = res.body?.getReader()
        if (!reader) throw new Error('No response body')
        const decoder = new TextDecoder()
        let fullContent = ''
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            const data = line.slice(6).trim()
            if (data === '[DONE]') continue
            try {
              const parsed = JSON.parse(data)
              if (parsed.error) { setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: parsed.error, isError: true } : m)); fullContent = parsed.error; break }
              if (parsed.model) continue
              const delta = parsed.choices?.[0]?.delta?.content
              if (delta) { fullContent += delta; setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: fullContent } : m)) }
            } catch { /* skip */ }
          }
        }
      } else {
        const data = await res.json()
        const reply = data.error || data.content || 'No response received.'
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: reply, isError: !!data.error } : m))
      }
    } catch (err: any) {
      if (err.name === 'AbortError') return
      setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: 'Network error. Check your connection and try again.', isError: true } : m))
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
      setMessages(currentMsgs => { persistMessages(currentMsgs, activeConvId, selectedModel); return currentMsgs })
    }
  }, [messages, selectedModel, temperature, maxTokens, systemPrompt, activeConvId, persistMessages])

  return (
    <div className="h-screen flex flex-col bg-background text-foreground">
      {/* ━━━ Top Bar ━━━ */}
      <header className="flex items-center justify-between h-14 px-4 border-b border-border/50 bg-background/80 backdrop-blur-xl shrink-0">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => setHistoryOpen(!historyOpen)} className="gap-1.5 text-muted-foreground hover:text-foreground">
            <MessageSquare className="w-4 h-4" />
            <span className="hidden sm:inline">History</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={startNewChat} className="gap-1.5 text-muted-foreground">
            <Plus className="w-4 h-4" />
            <span className="hidden sm:inline">New</span>
          </Button>
          <div className="hidden sm:block w-px h-6 bg-border" />
          <div className="hidden sm:flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold tracking-tight">Nexus<span className="text-teal-600">AI</span></span>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="sm" onClick={handleExport} disabled={messages.length === 0} className="gap-1.5 text-muted-foreground" title="Export chat">
            <FileDown className="w-4 h-4" />
            <span className="hidden sm:inline">Export</span>
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setSettingsOpen(!settingsOpen)} className="gap-1.5 text-muted-foreground">
            <Settings2 className="w-4 h-4" />
            <span className="hidden sm:inline">Settings</span>
          </Button>
          <Button variant="ghost" size="sm" asChild className="gap-1.5 text-muted-foreground hover:text-foreground">
            <Link href="/"><ArrowLeft className="w-4 h-4" /><span className="hidden sm:inline">Home</span></Link>
          </Button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* ━━━ History Sidebar ━━━ */}
        {historyOpen && (
          <aside className="w-64 shrink-0 border-r border-border/50 bg-muted/20 overflow-y-auto hidden md:block">
            <div className="p-3">
              <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2 px-1">Conversations</p>
              <div className="space-y-0.5">
                {conversations.length === 0 && (
                  <p className="text-xs text-muted-foreground text-center py-6">No conversations yet.<br/>Start chatting!</p>
                )}
                {conversations.map(conv => (
                  <div
                    key={conv.id}
                    onClick={() => loadConversation(conv)}
                    className={`group flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer text-sm transition-colors ${
                      activeConvId === conv.id ? 'bg-muted border border-border' : 'hover:bg-muted/50'
                    }`}
                  >
                    <MessageSquare className="w-3.5 h-3.5 shrink-0 text-muted-foreground" />
                    <div className="flex-1 min-w-0">
                      <p className="truncate text-xs font-medium">{conv.title}</p>
                      <p className="text-[10px] text-muted-foreground">{new Date(conv.updatedAt).toLocaleDateString()}</p>
                    </div>
                    <button onClick={(e) => handleDeleteConversation(conv.id, e)} className="opacity-0 group-hover:opacity-100 p-0.5 hover:text-red-500 transition-all">
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </aside>
        )}

        {/* ━━━ Settings Sidebar ━━━ */}
        {settingsOpen && (
          <aside className="w-72 shrink-0 border-r border-border/50 bg-muted/30 overflow-y-auto hidden md:block">
            <div className="p-4 space-y-5">
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">AI Model</label>
                <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500">
                  {MODEL_OPTIONS.map(m => (<option key={m} value={m}>{m}</option>))}
                </select>
              </div>
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">System Prompt</label>
                <textarea value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)} rows={3} className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500 resize-none" />
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Temperature</label>
                  <span className="text-xs text-teal-600 font-mono">{temperature}</span>
                </div>
                <input type="range" min="0.1" max="1.5" step="0.1" value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} className="w-full accent-teal-600" />
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Max Tokens</label>
                  <span className="text-xs text-teal-600 font-mono">{maxTokens}</span>
                </div>
                <input type="range" min="128" max="4096" step="128" value={maxTokens} onChange={e => setMaxTokens(parseInt(e.target.value))} className="w-full accent-teal-600" />
              </div>
              <hr className="border-border/50" />
              <div>
                <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">Quick Actions</label>
                <div className="grid grid-cols-2 gap-2">
                  {QUICK_ACTIONS.map(qa => (
                    <button key={qa.label} onClick={() => setPendingPrompt(qa.prompt)} disabled={isLoading} className="flex items-center gap-1.5 px-3 py-2 text-xs font-medium rounded-md border border-border bg-background hover:bg-muted transition-colors disabled:opacity-50">
                      {qa.icon} {qa.label}
                    </button>
                  ))}
                </div>
              </div>
              <hr className="border-border/50" />
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <Wrench className="w-3.5 h-3.5 text-muted-foreground" />
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Agent Tools</label>
                </div>
                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2"><Calculator className="w-3 h-3" /> Calculator</div>
                  <div className="flex items-center gap-2"><Terminal className="w-3 h-3" /> Code Executor</div>
                  <div className="flex items-center gap-2"><Clock className="w-3 h-3" /> Date/Time</div>
                  <div className="flex items-center gap-2"><Globe className="w-3 h-3" /> Web Search</div>
                </div>
              </div>
              <hr className="border-border/50" />
              <div>
                <div className="flex items-center gap-1.5 mb-2">
                  <FileText className="w-3.5 h-3.5 text-muted-foreground" />
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Resources</label>
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
          {/* Mobile bar */}
          <div className="md:hidden flex items-center gap-2 px-3 py-2 border-b border-border/50 overflow-x-auto">
            <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} className="h-8 rounded-md border border-input bg-background px-2 text-xs focus:outline-none">
              {MODEL_OPTIONS.map(m => (<option key={m} value={m}>{m.split('(')[0].trim()}</option>))}
            </select>
            {QUICK_ACTIONS.map(qa => (
              <button key={qa.label} onClick={() => setPendingPrompt(qa.prompt)} disabled={isLoading} className="shrink-0 px-2.5 py-1 text-xs font-medium rounded-full border border-border bg-background hover:bg-muted transition-colors disabled:opacity-50">
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
                  Free multi-model AI chat with code execution, web search, and more. Conversations are saved automatically.
                </p>
                <div className="flex flex-wrap justify-center gap-2 mt-6">
                  {QUICK_ACTIONS.map(qa => (
                    <button key={qa.label} onClick={() => setPendingPrompt(qa.prompt)} className="flex items-center gap-1.5 px-4 py-2 text-sm rounded-full border border-border bg-background hover:bg-muted transition-colors">
                      {qa.icon} {qa.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map(msg => (
              <div key={msg.id} className={`max-w-3xl mx-auto mb-4 flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                <div className={`shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                  msg.role === 'user' ? 'bg-teal-100 text-teal-700' : 'bg-gradient-to-br from-teal-500 to-cyan-600 text-white'
                }`}>
                  {msg.role === 'user' ? <span className="text-sm font-bold">O</span> : <Bot className="w-4 h-4" />}
                </div>
                <ChatMessageBubble
                  content={msg.content}
                  role={msg.role as 'user' | 'assistant'}
                  isError={msg.isError}
                  onRetry={msg.isError ? handleRetry : undefined}
                />
              </div>
            ))}

            {isLoading && messages[messages.length - 1]?.content === '' && (
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
              {/* File upload button */}
              <input ref={fileInputRef} type="file" accept=".txt,.md,.csv,.json,.js,.ts,.py,.html,.css,.xml,.yaml,.yml,.log" onChange={handleFileUpload} className="hidden" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                className="h-11 w-11 rounded-xl p-0 shrink-0 text-muted-foreground hover:text-foreground"
                title="Upload file"
              >
                <Upload className="w-5 h-5" />
              </Button>

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
              {isLoading ? (
                <Button onClick={stopGeneration} className="h-11 w-11 rounded-xl bg-red-500 hover:bg-red-600 text-white p-0 shrink-0" title="Stop generation">
                  <RefreshCw className="w-5 h-5" />
                </Button>
              ) : (
                <Button onClick={() => handleSendMessage()} disabled={!input.trim() && !attachedFile} className="h-11 w-11 rounded-xl bg-teal-600 hover:bg-teal-700 text-white p-0 shrink-0">
                  <Send className="w-5 h-5" />
                </Button>
              )}
            </div>

            {/* File attachment indicator */}
            {attachedFile && (
              <div className="max-w-3xl mx-auto mt-2 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted border border-border text-xs">
                <FileText className="w-3.5 h-3.5 text-teal-600" />
                <span className="truncate flex-1">{attachedFile.name}</span>
                <button onClick={() => setAttachedFile(null)} className="text-muted-foreground hover:text-foreground">x</button>
              </div>
            )}

            <p className="text-center text-[11px] text-muted-foreground mt-2">
              Streaming responses | Code execution via Piston API | Conversations saved locally
            </p>
          </div>
        </main>
      </div>
    </div>
  )
}
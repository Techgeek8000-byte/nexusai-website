'use client'

import { useState, useRef, useEffect, useCallback, Suspense } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import {
  Bot, Send, Trash2, ArrowLeft, Settings2, Code2, Search, Hash, Calculator,
  Loader2, ChevronLeft, ChevronRight, Sparkles, Wrench, FileText, Download,
  Plus, MessageSquare, Clock, Upload, FileDown, RefreshCw, Terminal, Globe, Cpu, Lightbulb, BookOpen, PenTool
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ChatMessageBubble } from '@/components/chat-message'
import { ThemeToggle } from '@/components/theme-toggle'
import { detectToolsClient } from '@/lib/tools'
import {
  rateResponse as rateResponseStore,
  getAdaptivePrompt,
  compressContext,
  getInsights,
  resetLearning,
  type LearningInsights,
} from '@/lib/self-improve'
import {
  type SavedConversation, loadConversations, saveConversation,
  deleteConversation, getActiveConversationId, setActiveConversationId,
  generateId, generateTitle
} from '@/lib/storage'

// ━━━ Types ━━━
interface AnalysisInfo {
  improved: boolean
  quality_score: number
  issues: { type: string; severity: string; description: string }[]
  analysis_time_ms: number
  cache_hit?: boolean
}

interface ChatMessage {
  id: number
  role: 'user' | 'assistant' | 'system'
  content: string
  isError?: boolean
  model?: string
  analysisInfo?: AnalysisInfo | null
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

const EMPTY_STATE_PROMPTS = [
  { icon: <Lightbulb className="w-4 h-4" />, text: 'Explain quantum computing in simple terms' },
  { icon: <Code2 className="w-4 h-4" />, text: 'Write a Python web scraper' },
  { icon: <Globe className="w-4 h-4" />, text: 'What is the latest AI news?' },
  { icon: <BookOpen className="w-4 h-4" />, text: 'Summarize the history of the internet' },
  { icon: <PenTool className="w-4 h-4" />, text: 'Write a professional email to a client' },
  { icon: <Terminal className="w-4 h-4" />, text: 'Run JavaScript code to sort an array' },
]

const DEFAULT_SYSTEM_PROMPT = (
  'You are NexusAI, a helpful, knowledgeable AI assistant created by Osama. '
  + 'You respond fluently in both English and Urdu. '
  + 'Be concise, accurate, and friendly. When writing code, include explanations. '
  + 'When the user asks you to execute or run code, use the execute_code tool. '
  + 'When the user asks about current information, use the web_search tool.'
)

// ━━━ Main Component ━━━
function ChatPage() {
  const searchParams = useSearchParams()
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
  const [insights, setInsights] = useState<LearningInsights | null>(null)
  const [responseModel, setResponseModel] = useState<string>('')
  const ratingsRef = useRef<Map<number, 1 | -1>>(new Map())
  const analysisRef = useRef<Map<number, AnalysisInfo>>(new Map())
  const [cloudStats, setCloudStats] = useState<{ enabled: boolean; totalImprovements?: number; totalIssuesFixed?: number; cacheHits?: number; cacheSize?: number } | null>(null)

  // ── Handle ?prompt= from landing page ──
  useEffect(() => {
    const urlPrompt = searchParams.get('prompt')
    if (urlPrompt) {
      setPendingPrompt(urlPrompt)
    }
  }, [searchParams])

  // ── Load insights for settings panel ──
  useEffect(() => {
    if (settingsOpen) {
      setInsights(getInsights())
      fetch('/api/cache-stats').then(r => r.json()).then(setCloudStats).catch(() => {})
    }
  }, [settingsOpen])

  // ── Handle rate response ──
  const handleRate = useCallback((msgId: number, rating: 1 | -1) => {
    const msg = messages.find(m => m.id === msgId)
    if (!msg || msg.role !== 'assistant') return
    const lastUserMsg = [...messages].reverse().find(m => m.role === 'user')
    rateResponseStore(lastUserMsg?.content || '', msg.content, rating, selectedModel)
    ratingsRef.current.set(msgId, rating)
    setInsights(getInsights())
  }, [messages, selectedModel])

  // ── Handle self-analysis events from SSE ──
  const handleAnalysisEvent = useCallback((msgId: number, event: any) => {
    if (event.type === 'self_improved') {
      // Replace the streamed content with the improved version
      const analysisInfo: AnalysisInfo = {
        improved: true,
        quality_score: event.quality_score,
        issues: event.issues || [],
        analysis_time_ms: event.analysis_time_ms,
      }
      analysisRef.current.set(msgId, analysisInfo)
      setMessages(prev => prev.map(m =>
        m.id === msgId
          ? { ...m, content: event.improved_response, analysisInfo }
          : m
      ))
    } else if (event.type === 'self_analysis') {
      // Response passed review — show quality badge
      const analysisInfo: AnalysisInfo = {
        improved: false,
        quality_score: event.quality_score,
        issues: event.issues || [],
        analysis_time_ms: event.analysis_time_ms,
      }
      analysisRef.current.set(msgId, analysisInfo)
      setMessages(prev => prev.map(m =>
        m.id === msgId ? { ...m, analysisInfo } : m
      ))
    } else if (event.type === 'cache_hit') {
      const analysisInfo: AnalysisInfo = {
        improved: false,
        quality_score: event.quality_score || 90,
        issues: [],
        analysis_time_ms: 0,
        cache_hit: true,
      }
      analysisRef.current.set(msgId, analysisInfo)
      setMessages(prev => prev.map(m =>
        m.id === msgId ? { ...m, analysisInfo } : m
      ))
    }
  }, [])

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

    // Build API messages with adaptive prompt + context compression
    const adaptivePrompt = getAdaptivePrompt(systemPrompt)
    const rawApiMessages = [
      { role: 'system' as const, content: adaptivePrompt },
      ...updatedMessages.filter(m => m.role !== 'system').map(m => ({ role: m.role as 'user' | 'assistant', content: m.content })),
    ]
    // Replace last user message with full content if file was attached
    if (attachedFile || (fullContent !== messageText)) {
      const lastUserIdx = rawApiMessages.length - 1
      if (rawApiMessages[lastUserIdx].role === 'user') {
        rawApiMessages[lastUserIdx].content = fullContent
      }
    }
    // Compress long conversations
    const { messages: apiMessages } = compressContext(rawApiMessages)

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
              if (parsed.model) { setResponseModel(parsed.model); continue }
              // Handle self-analysis events (sent after stream completes)
              if (parsed.type === 'self_improved' || parsed.type === 'self_analysis' || parsed.type === 'cache_hit') {
                handleAnalysisEvent(assistantId, parsed)
                continue
              }
              const delta = parsed.choices?.[0]?.delta?.content
              if (delta) {
                fullContent += delta
                setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: fullContent, model: responseModel } : m))
              }
            } catch { /* skip */ }
          }
        }
      } else {
        // ── Non-streaming JSON fallback ──
        const data = await res.json()
        const reply = data.error || data.content || 'No response received.'
        const isErr = !!data.error
        // Handle self-analysis from non-streaming response
        const analysisInfo = data.self_analysis ? {
          improved: data.self_analysis.improved,
          quality_score: data.self_analysis.quality_score,
          issues: data.self_analysis.issues || [],
          analysis_time_ms: data.self_analysis.analysis_time_ms,
          cache_hit: data.cache_hit,
        } : (data.cache_hit ? {
          improved: false,
          quality_score: data.quality_score || 90,
          issues: [],
          analysis_time_ms: 0,
          cache_hit: true,
        } : null)
        if (analysisInfo) analysisRef.current.set(assistantId, analysisInfo)
        setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: reply, isError: isErr, analysisInfo } : m))
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
              if (parsed.model) { setResponseModel(parsed.model); continue }
              if (parsed.type === 'self_improved' || parsed.type === 'self_analysis' || parsed.type === 'cache_hit') {
                handleAnalysisEvent(assistantId, parsed)
                continue
              }
              const delta = parsed.choices?.[0]?.delta?.content
              if (delta) { fullContent += delta; setMessages(prev => prev.map(m => m.id === assistantId ? { ...m, content: fullContent, model: responseModel } : m)) }
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
          <ThemeToggle />
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
                <div className="flex items-center gap-1.5 mb-3">
                  <Sparkles className="w-3.5 h-3.5 text-teal-600" />
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Self-Improvement</label>
                </div>
                <div className="space-y-3">
                  {/* Cloud Self-Analysis */}
                  <div className="p-2.5 rounded-lg bg-gradient-to-br from-violet-50 to-teal-50 dark:from-violet-950/20 dark:to-teal-950/20 border border-violet-200/50 dark:border-violet-800/30">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-violet-600 dark:text-violet-400 mb-1.5">Cloud Auto-Analysis</p>
                    <p className="text-[10px] text-muted-foreground leading-relaxed mb-2">
                      Every response is automatically reviewed for code errors, security vulnerabilities, and quality. Issues are auto-fixed and the improved version replaces the weaker one.
                    </p>
                    {cloudStats && cloudStats.enabled ? (
                      <div className="space-y-1.5 text-[10px]">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Responses improved</span>
                          <span className="font-mono text-violet-600 dark:text-violet-400">{cloudStats.totalImprovements || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Issues fixed</span>
                          <span className="font-mono text-emerald-600">{cloudStats.totalIssuesFixed || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Cache hits</span>
                          <span className="font-mono text-teal-600">{cloudStats.cacheHits || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Cached answers</span>
                          <span className="font-mono text-foreground">{cloudStats.cacheSize || 0}</span>
                        </div>
                        <button
                          onClick={async () => {
                            const res = await fetch('/api/cache-stats', { method: 'POST' })
                            const data = await res.json()
                            fetch('/api/cache-stats').then(r => r.json()).then(setCloudStats)
                            alert(`Cleaned up ${data.discarded} weak responses from cloud cache.`)
                          }}
                          className="w-full mt-1 px-2 py-1 text-[10px] rounded border border-violet-200 dark:border-violet-800 text-violet-600 hover:bg-violet-100 dark:hover:bg-violet-900/20 transition-colors"
                        >
                          Discard weak cached responses
                        </button>
                      </div>
                    ) : (
                      <p className="text-[10px] text-amber-600 dark:text-amber-400">
                        KV not connected — self-analysis still works, but improved responses aren't cached across sessions. Add Vercel KV (free) to enable cloud learning.
                      </p>
                    )}
                  </div>
                  {/* Local Feedback Learning */}
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">Local Feedback Learning</p>
                    {insights && insights.totalRated > 0 ? (
                      <div className="space-y-2 text-xs">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Feedback given</span>
                          <span className="font-mono text-foreground">{insights.totalRated} ratings</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Accuracy</span>
                          <span className="font-mono text-emerald-600">{Math.round(insights.avgRating * 100)}%</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Learned style</span>
                          <span className="font-mono capitalize text-foreground">{insights.stylePreference}</span>
                        </div>
                        {insights.totalSavedTokens > 0 && (
                          <div className="flex justify-between items-center">
                            <span className="text-muted-foreground">Tokens saved</span>
                            <span className="font-mono text-teal-600">~{insights.totalSavedTokens.toLocaleString()}</span>
                          </div>
                        )}
                        {insights.topCategories.length > 0 && (
                          <div className="mt-1">
                            <span className="text-muted-foreground block mb-1">Strongest topics</span>
                            <div className="flex flex-wrap gap-1">
                              {insights.topCategories.slice(0, 3).map(c => (
                                <span key={c.category} className="px-2 py-0.5 rounded-full bg-teal-50 dark:bg-teal-900/20 text-teal-700 dark:text-teal-400 text-[10px] capitalize">
                                  {c.category} {Math.round(c.score * 100)}%
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        <button
                          onClick={() => { resetLearning(); setInsights(getInsights()); ratingsRef.current.clear() }}
                          className="w-full mt-2 px-3 py-1.5 text-xs rounded-md border border-red-200 dark:border-red-800 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                        >
                          Reset learning data
                        </button>
                      </div>
                    ) : (
                      <p className="text-xs text-muted-foreground leading-relaxed">
                        Rate responses with thumbs up/down to teach NexusAI your preferences. It adapts its style, context, and prompts based on your feedback.
                      </p>
                    )}
                  </div>
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
              <div className="h-full flex flex-col items-center justify-center text-center px-4">
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center mb-4 shadow-lg shadow-teal-500/20">
                  <Bot className="w-8 h-8 text-white" />
                </div>
                <h2 className="text-xl font-bold">What can I help with?</h2>
                <p className="text-sm text-muted-foreground mt-2 max-w-md">
                  Free multi-model AI chat with code execution, web search, file analysis, and more. Pick a suggestion or type anything below.
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-8 max-w-lg w-full">
                  {EMPTY_STATE_PROMPTS.map((p, i) => (
                    <button
                      key={i}
                      onClick={() => setPendingPrompt(p.text)}
                      disabled={isLoading}
                      className="flex items-center gap-3 px-4 py-3 text-sm text-left rounded-xl border border-border/50 bg-card hover:bg-muted hover:border-teal-200 dark:hover:border-teal-800 transition-all disabled:opacity-50 group"
                    >
                      <span className="text-muted-foreground group-hover:text-teal-600 transition-colors">{p.icon}</span>
                      <span className="text-muted-foreground group-hover:text-foreground transition-colors">{p.text}</span>
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
                  messageId={msg.id}
                  showRating={msg.role === 'assistant' && !msg.isError}
                  onRate={(r) => handleRate(msg.id, r)}
                  currentRating={ratingsRef.current.get(msg.id) ?? null}
                  model={msg.model}
                  analysisInfo={msg.analysisInfo}
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
              NexusAI &middot; {selectedModel.split('(')[0].trim()} &middot; Free &middot; No signup
            </p>
          </div>
        </main>
      </div>
    </div>
  )
}

function ChatLoadingFallback() {
  return (
    <div className="flex h-screen items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-3">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <p className="text-sm text-muted-foreground">Loading NexusAI...</p>
      </div>
    </div>
  )
}

export default function ChatPageWrapper() {
  return (
    <Suspense fallback={<ChatLoadingFallback />}>
      <ChatPage />
    </Suspense>
  )
}
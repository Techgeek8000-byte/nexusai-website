'use client'

import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import {
  Bot, Zap, Globe, Shield, Code2, BrainCircuit,
  ArrowRight, Check, ChevronDown, Menu, X,
  Github, ExternalLink, MessageSquare, Layers,
  Cpu, Network, Sparkles, Lock
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

// ━━━ Chat Demo Messages ━━━
interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
}

const DEMO_RESPONSES: Record<string, string> = {
  hello: "Hi! This is a frontend demo of NexusAI. I can show you what the real chat at /chat does — multi-model conversations, code execution, web search, streaming responses, and file upload. Try asking about code, search, models, or files!",
  code: "In the real NexusAI chat, you can write and execute code in 20+ languages using the Piston API. For example, send a Python snippet and it runs in a sandboxed environment. Try it at /chat!",
  search: "NexusAI has web search powered by Tavily, with Wikipedia as a fallback. Ask the real AI anything current — it'll search the web and give you sourced answers. Head to /chat to try it.",
  model: "NexusAI supports 4 Qwen 2.5 models (0.5B, 1.5B, 3B, 7B) that you can switch between in the chat. The API at /v1/chat/completions is OpenAI-compatible with streaming. Visit /chat to pick a model.",
  file: "You can upload files directly in the NexusAI chat. The AI reads and analyzes them. Try uploading a .txt, .py, .json, or .md file at /chat.",
  default: "This is a frontend demo — the responses are pre-written. For the real AI with multi-model chat, code execution, web search, and streaming, open the full chat at /chat."
}

function getDemoResponse(input: string): string {
  const lower = input.toLowerCase()
  if (lower.includes('hello') || lower.includes('hi') || lower.includes('hey')) return DEMO_RESPONSES.hello
  if (lower.includes('code') || lower.includes('python') || lower.includes('program') || lower.includes('execute') || lower.includes('run')) return DEMO_RESPONSES.code
  if (lower.includes('search') || lower.includes('web') || lower.includes('browse') || lower.includes('lookup')) return DEMO_RESPONSES.search
  if (lower.includes('model') || lower.includes('qwen') || lower.includes('switch') || lower.includes('stream')) return DEMO_RESPONSES.model
  if (lower.includes('file') || lower.includes('upload') || lower.includes('document')) return DEMO_RESPONSES.file
  return DEMO_RESPONSES.default
}

// ━━━ Feature Data ━━━
const FEATURES = [
  {
    icon: <BrainCircuit className="w-6 h-6" />,
    title: "Multi-Model Chat",
    description: "Choose from 4 Qwen 2.5 models (0.5B, 1.5B, 3B, 7B) in the chat interface. Switch between them depending on your task.",
    tag: "Live"
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: "Streaming Responses",
    description: "Real-time token-by-token streaming via SSE. Watch the AI respond live, just like ChatGPT. Fast and responsive.",
    tag: "Live"
  },
  {
    icon: <Code2 className="w-6 h-6" />,
    title: "Code Execution",
    description: "Write and run code in 20+ languages using the Piston API. Sandbox-isolated execution with instant results.",
    tag: "Live"
  },
  {
    icon: <Globe className="w-6 h-6" />,
    title: "Web Search",
    description: "Real-time web search powered by Tavily, with Wikipedia as a fallback. Get up-to-date answers with sources.",
    tag: "Live"
  },
  {
    icon: <Network className="w-6 h-6" />,
    title: "OpenAI-Compatible API",
    description: "Use /v1/chat/completions with any OpenAI SDK. Switch the base URL and it works. Streaming included.",
    tag: "Live"
  },
  {
    icon: <Shield className="w-6 h-6" />,
    title: "100% Free",
    description: "No cost, no API key required, no subscriptions. Powered by free-tier infrastructure and open-source models.",
    tag: "Free"
  }
]

const STATS = [
  { value: "$0", label: "Cost", desc: "Completely free to use" },
  { value: "4", label: "AI Models", desc: "Qwen 2.5 family" },
  { value: "4", label: "Agent Tools", desc: "Calculator, datetime, code, search" },
  { value: "2", label: "Languages", desc: "English and Urdu support" }
]

const API_ENDPOINTS = [
  { method: 'POST', path: '/v1/chat/completions', desc: 'Chat completions with streaming' },
]

// ━━━ Main Component ━━━
export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, role: 'assistant', content: "Hi! This is a frontend demo — responses are pre-written. For the real AI with multi-model chat, code execution, web search, and streaming, open the full chat at /chat." }
  ])
  const [input, setInput] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const chatEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    if (!input.trim()) return
    const userMsg: Message = { id: Date.now(), role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setIsTyping(true)
    setInput('')
    setTimeout(() => {
      const response = getDemoResponse(input)
      setMessages(prev => [...prev, { id: Date.now() + 1, role: 'assistant', content: response }])
      setIsTyping(false)
    }, 800 + Math.random() * 700)
  }

  return (
    <div className="min-h-screen flex flex-col bg-background">
      {/* ━━━ Navigation ━━━ */}
      <header className="sticky top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <span className="text-lg font-bold tracking-tight">
                Nexus<span className="text-teal-600">AI</span>
              </span>
            </div>

            {/* Desktop Nav */}
            <nav className="hidden md:flex items-center gap-8">
              <Link href="/chat" className="text-sm text-teal-600 hover:text-teal-700 font-medium transition-colors">
                Chat
              </Link>
              {['Features', 'Demo', 'API', 'Roadmap'].map(item => (
                <a key={item} href={`#${item.toLowerCase()}`}
                   className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                  {item}
                </a>
              ))}
            </nav>

            <div className="hidden md:flex items-center gap-3">
              <Button variant="ghost" size="sm" asChild>
                <a href="#api" className="gap-2"><Github className="w-4 h-4" /> GitHub</a>
              </Button>
              <Button size="sm" className="bg-teal-600 hover:bg-teal-700 text-white gap-2" asChild>
                <Link href="/chat">Open Chat <ArrowRight className="w-4 h-4" /></Link>
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <button className="md:hidden p-2" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden border-t border-border/50 py-4 space-y-3">
              {['Features', 'Demo', 'API', 'Roadmap'].map(item => (
                <a key={item} href={`#${item.toLowerCase()}`}
                   className="block text-sm text-muted-foreground hover:text-foreground py-2"
                   onClick={() => setMobileMenuOpen(false)}>
                  {item}
                </a>
              ))}
              <Button size="sm" className="w-full bg-teal-600 hover:bg-teal-700 text-white gap-2" asChild>
                <Link href="/chat">Open Chat <ArrowRight className="w-4 h-4" /></Link>
              </Button>
            </div>
          )}
        </div>
      </header>

      <main className="flex-1">
        {/* ━━━ Hero Section ━━━ */}
        <section className="relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-b from-teal-50/50 to-transparent" />
          <div className="absolute top-20 left-1/4 w-72 h-72 bg-teal-200/20 rounded-full blur-3xl" />
          <div className="absolute top-40 right-1/4 w-96 h-96 bg-cyan-200/15 rounded-full blur-3xl" />

          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-24 sm:pt-28 sm:pb-32">
            <div className="text-center max-w-4xl mx-auto">
              <Badge variant="secondary" className="mb-6 px-4 py-1.5 text-sm bg-teal-100 text-teal-800 border-teal-200">
                A Project by Osama
              </Badge>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight leading-tight">
                Build AI That{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-cyan-600">
                  Actually Does Things
                </span>
              </h1>

              <p className="mt-6 text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
                NexusAI is a free AI platform with multi-model chat, code execution, web search, and streaming responses. Powered by Qwen 2.5.
              </p>

              <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button size="lg" className="bg-teal-600 hover:bg-teal-700 text-white px-8 h-12 text-base gap-2" asChild>
                  <Link href="/chat"><Zap className="w-5 h-5" /> Open AI Chat</Link>
                </Button>
                <Button size="lg" variant="outline" className="px-8 h-12 text-base gap-2" asChild>
                  <a href="#demo"><MessageSquare className="w-5 h-5" /> Try Live Demo</a>
                </Button>
              </div>
            </div>

            {/* Stats Bar */}
            <div className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-6 max-w-3xl mx-auto">
              {STATS.map(stat => (
                <div key={stat.label} className="text-center">
                  <div className="text-3xl sm:text-4xl font-extrabold text-teal-600">{stat.value}</div>
                  <div className="mt-1 text-sm font-semibold">{stat.label}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">{stat.desc}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ━━━ Features Section ━━━ */}
        <section id="features" className="py-20 sm:py-28">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-2xl mx-auto mb-16">
              <Badge variant="secondary" className="mb-4">Platform Features</Badge>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
                Real Features, Not Promises
              </h2>
              <p className="mt-4 text-muted-foreground text-lg">
                From intelligent chat to autonomous agents, NexusAI provides a complete platform for building AI-powered applications.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {FEATURES.map(feature => (
                <Card key={feature.title} className="group hover:shadow-lg transition-shadow duration-300 border-border/50">
                  <CardHeader>
                    <div className="flex items-center justify-between mb-2">
                      <div className="w-12 h-12 rounded-xl bg-teal-50 flex items-center justify-center text-teal-600 group-hover:bg-teal-100 transition-colors">
                        {feature.icon}
                      </div>
                      <Badge variant="outline" className="text-xs">{feature.tag}</Badge>
                    </div>
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-sm leading-relaxed">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        {/* ━━━ Interactive Demo Section ━━━ */}
        <section id="demo" className="py-20 sm:py-28 bg-gradient-to-b from-slate-50 to-background">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-2xl mx-auto mb-16">
              <Badge variant="secondary" className="mb-4">Live Demo</Badge>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
                Try NexusAI Right Now
              </h2>
              <p className="mt-4 text-muted-foreground text-lg">
                This is a frontend demo. For the real AI, open the full chat.
              </p>
            </div>

            <div className="max-w-3xl mx-auto">
              <Card className="border-2 border-teal-200/50 shadow-xl">
                <CardHeader className="border-b bg-gradient-to-r from-teal-50 to-cyan-50/50">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center">
                      <Bot className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-base">NexusAI Chat</CardTitle>
                      <CardDescription className="text-xs">Powered by Qwen 2.5 7B</CardDescription>
                    </div>
                    <Badge className="ml-auto bg-emerald-100 text-emerald-700">Online</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Chat Messages */}
                  <div className="h-[400px] overflow-y-auto space-y-4 p-4 mb-4 bg-slate-50/50 rounded-lg">
                    {messages.map(msg => (
                      <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                          msg.role === 'assistant'
                            ? 'bg-gradient-to-br from-teal-500 to-cyan-600'
                            : 'bg-slate-700'
                        }`}>
                          {msg.role === 'assistant'
                            ? <Bot className="w-4 h-4 text-white" />
                            : <span className="text-xs text-white font-bold">U</span>
                          }
                        </div>
                        <div className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                          msg.role === 'assistant'
                            ? 'bg-white border border-border/50 shadow-sm'
                            : 'bg-teal-600 text-white'
                        }`}>
                          <div className="whitespace-pre-wrap">{msg.content}</div>
                        </div>
                      </div>
                    ))}
                    {isTyping && (
                      <div className="flex gap-3">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center">
                          <Bot className="w-4 h-4 text-white" />
                        </div>
                        <div className="bg-white border border-border/50 rounded-2xl px-4 py-3 shadow-sm">
                          <div className="flex gap-1.5">
                            <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                            <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                            <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </div>

                  {/* Quick Prompts */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {['Write Python code', 'Web search', 'What models are available?', 'Upload a file'].map(prompt => (
                      <Button
                        key={prompt}
                        variant="outline"
                        size="sm"
                        className="text-xs h-8"
                        onClick={() => { setInput(prompt) }}
                      >
                        {prompt}
                      </Button>
                    ))}
                  </div>

                  {/* Input */}
                  <div className="flex gap-2">
                    <Input
                      placeholder="Ask NexusAI anything..."
                      value={input}
                      onChange={e => setInput(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleSend()}
                      className="flex-1"
                    />
                    <Button onClick={handleSend} className="bg-teal-600 hover:bg-teal-700 text-white" disabled={!input.trim()}>
                      <ArrowRight className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>

        {/* ━━━ API Documentation Section ━━━ */}
        <section id="api" className="py-20 sm:py-28">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-2xl mx-auto mb-16">
              <Badge variant="secondary" className="mb-4">Developer API</Badge>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
                OpenAI-Compatible API
              </h2>
              <p className="mt-4 text-muted-foreground text-lg">
                Switch from OpenAI with one line. Streaming included.
              </p>
            </div>

            <Tabs defaultValue="endpoints" className="max-w-4xl mx-auto">
              <TabsList className="grid w-full grid-cols-2 mb-8">
                <TabsTrigger value="endpoints">Endpoints</TabsTrigger>
                <TabsTrigger value="quickstart">Quick Start</TabsTrigger>
              </TabsList>

              <TabsContent value="endpoints">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Network className="w-5 h-5 text-teal-600" /> API Endpoints
                    </CardTitle>
                    <CardDescription>OpenAI-compatible endpoint</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {API_ENDPOINTS.map(ep => (
                        <div key={ep.path} className="flex items-center gap-3 p-3 bg-slate-50 rounded-lg">
                          <code className={`text-xs font-bold px-2 py-1 rounded ${
                            ep.method === 'GET' ? 'bg-emerald-100 text-emerald-700' : 'bg-blue-100 text-blue-700'
                          }`}>{ep.method}</code>
                          <code className="text-sm font-mono flex-1">{ep.path}</code>
                          <span className="text-xs text-muted-foreground hidden sm:block">{ep.desc}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="quickstart">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Code2 className="w-5 h-5 text-teal-600" /> Quick Start
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="p-4 bg-slate-900 rounded-lg overflow-x-auto">
                        <pre className="text-sm text-green-400 font-mono whitespace-pre">
{`# Install the SDK
pip install openai

# Switch from OpenAI to NexusAI (one line!)
openai.base_url = "https://your-vercel-url.vercel.app/v1"
openai.api_key = "any-value"

# Chat completion - works exactly like OpenAI
response = openai.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "Hello NexusAI!"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")`}
                        </pre>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Badge variant="outline" className="text-xs"><Check className="w-3 h-3 mr-1" /> Drop-in replacement</Badge>
                        <Badge variant="outline" className="text-xs"><Check className="w-3 h-3 mr-1" /> Streaming support</Badge>
                        <Badge variant="outline" className="text-xs"><Check className="w-3 h-3 mr-1" /> No API key needed</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </section>

        {/* ━━━ Roadmap Section ━━━ */}
        <section id="roadmap" className="py-20 sm:py-28 bg-gradient-to-b from-slate-50 to-background">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-2xl mx-auto mb-16">
              <Badge variant="secondary" className="mb-4">Development Roadmap</Badge>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
                Building the Future, One Phase at a Time
              </h2>
            </div>

            <div className="max-w-3xl mx-auto space-y-6">
              {[
                { phase: 'Phase 1', title: 'Foundation', timeline: 'Done', items: ['Multi-model chat (4 Qwen models)', 'Streaming responses (SSE)', 'Conversation history', 'OpenAI-compatible API endpoint', 'Rate limit with fallback'], status: 'current' },
                { phase: 'Phase 2', title: 'Agent Tools', timeline: 'Done', items: ['Calculator tool', 'Datetime tool', 'Code execution (Piston API)', 'Web search (Tavily + Wikipedia)', 'File upload and analysis'], status: 'current' },
                { phase: 'Phase 3', title: 'UX Polish', timeline: 'Next', items: ['User accounts', 'Cloud conversation sync', 'Syntax highlighting', 'Dark theme', 'PWA support'], status: 'upcoming' },
                { phase: 'Phase 4', title: 'Custom Models', timeline: 'Future', items: ['Fine-tuned models', 'Urdu-optimized model', 'Domain-specific models', 'Model evaluation benchmarks', 'Community model sharing'], status: 'upcoming' }
              ].map((phase, idx) => (
                <Card key={phase.phase} className="border-border/50">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold ${
                          phase.status === 'current' ? 'bg-teal-600 text-white' : 'bg-slate-200 text-slate-600'
                        }`}>
                          {idx + 1}
                        </div>
                        <div>
                          <CardTitle className="text-base">{phase.phase}: {phase.title}</CardTitle>
                          <CardDescription>{phase.timeline}</CardDescription>
                        </div>
                      </div>
                      {phase.status === 'current' && (
                        <Badge className="bg-teal-100 text-teal-700">Done</Badge>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <ul className="grid sm:grid-cols-2 gap-2">
                      {phase.items.map(item => (
                        <li key={item} className="flex items-center gap-2 text-sm">
                          <Check className="w-4 h-4 text-teal-600 flex-shrink-0" /> {item}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        {/* ━━━ CTA Section ━━━ */}
        <section className="py-20 sm:py-28">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <div className="max-w-2xl mx-auto">
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
                Ready to Build the Future of AI?
              </h2>
              <p className="mt-4 text-lg text-muted-foreground">
                Join the NexusAI community. Start building for free today and be part of the open AI revolution.
              </p>
              <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button size="lg" className="bg-teal-600 hover:bg-teal-700 text-white px-8 h-12 text-base gap-2">
                  <Zap className="w-5 h-5" /> Get Started Free
                </Button>
                <Button size="lg" variant="outline" className="px-8 h-12 text-base gap-2" asChild>
                  <a href="#features"><ExternalLink className="w-5 h-5" /> View Documentation</a>
                </Button>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* ━━━ Footer ━━━ */}
      <footer className="border-t border-border/50 bg-slate-50/50 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="md:col-span-2">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <span className="text-lg font-bold">Nexus<span className="text-teal-600">AI</span></span>
              </div>
              <p className="text-sm text-muted-foreground max-w-sm leading-relaxed">
                A free AI platform with multi-model chat, code execution, web search, and streaming responses. Powered by Qwen 2.5 on free-tier infrastructure.
              </p>
              <p className="text-xs text-muted-foreground mt-4 font-medium">A Project by Osama</p>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-3">Platform</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><a href="#features" className="hover:text-foreground transition-colors">Features</a></li>
                <li><a href="#api" className="hover:text-foreground transition-colors">API Documentation</a></li>
                <li><a href="#demo" className="hover:text-foreground transition-colors">Live Demo</a></li>
                <li><a href="#roadmap" className="hover:text-foreground transition-colors">Roadmap</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-sm mb-3">Resources</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><a href="#" className="hover:text-foreground transition-colors flex items-center gap-1"><Github className="w-3 h-3" /> GitHub</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors flex items-center gap-1"><ExternalLink className="w-3 h-3" /> Documentation</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Changelog</a></li>
                <li><a href="#" className="hover:text-foreground transition-colors">Community</a></li>
              </ul>
            </div>
          </div>
          <div className="mt-12 pt-6 border-t border-border/50 flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-xs text-muted-foreground">
              NexusAI - Free AI Platform. Open source.
            </p>
            <p className="text-xs text-muted-foreground">
              Built with love by Osama
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

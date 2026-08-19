'use client'

import { useState } from 'react'
import Link from 'next/link'
import {
  Bot, Zap, Globe, Shield, Code2, BrainCircuit,
  Check, Github, MessageSquare, ArrowRight,
  Cpu, Network, Sparkles, Terminal,
  Play, Menu, X, FileText, Upload
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ThemeToggle } from '@/components/theme-toggle'

// ━━━ Suggested Prompts for Empty State ━━━
const SUGGESTED_PROMPTS = [
  'Explain quantum computing in simple terms',
  'Write a Python web scraper',
  'What is the latest AI news?',
  'Calculate: (15^3) * 2 - 450',
]

// ━━━ Feature Data (all verified real) ━━━
const FEATURES = [
  {
    icon: <BrainCircuit className="w-6 h-6" />,
    title: "Multi-Model Chat",
    description: "Switch between 4 Qwen 2.5 models (0.5B to 7B) mid-conversation. Each model has different speed and quality tradeoffs. Automatic fallback if a model is rate-limited.",
    tag: "Live",
  },
  {
    icon: <Zap className="w-6 h-6" />,
    title: "Real-Time Streaming",
    description: "Token-by-token SSE streaming, just like ChatGPT. See the AI think in real time. Responses start appearing instantly, not after a long wait.",
    tag: "Live",
  },
  {
    icon: <Terminal className="w-6 h-6" />,
    title: "Code Execution",
    description: "Run code in 20+ languages (Python, JS, Go, Rust, C++, Java and more) in a sandboxed environment via Piston API. Get instant stdout and stderr.",
    tag: "Live",
  },
  {
    icon: <Globe className="w-6 h-6" />,
    title: "Web Search",
    description: "Search the web with Tavily, with Wikipedia as a free fallback. Get sourced, up-to-date answers. The AI decides when to search based on your question.",
    tag: "Live",
  },
  {
    icon: <Network className="w-6 h-6" />,
    title: "OpenAI-Compatible API",
    description: "Drop-in replacement for OpenAI at /v1/chat/completions. Use any OpenAI SDK by changing the base URL. Streaming included.",
    tag: "Live",
  },
  {
    icon: <Shield className="w-6 h-6" />,
    title: "100% Free, No Signup",
    description: "No account needed, no API key, no credit card, no subscriptions. Open the chat and start talking. Powered by free-tier infrastructure.",
    tag: "Free",
  },
]

// Honest stats - all verifiable
const STATS = [
  { value: "$0", label: "Cost", desc: "No signup, no payment" },
  { value: "4", label: "AI Models", desc: "Qwen 2.5 family (0.5B-7B)" },
  { value: "4", label: "Agent Tools", desc: "Calculator, code, search, time" },
  { value: "20+", label: "Code Languages", desc: "Sandboxed execution" },
]

const API_ENDPOINTS = [
  { method: 'POST', path: '/v1/chat/completions', desc: 'OpenAI-compatible chat with streaming' },
  { method: 'POST', path: '/api/chat', desc: 'Native chat with tools + fallback' },
]

// ━━━ Main Component ━━━
export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

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
            <nav className="hidden md:flex items-center gap-6">
              <Link href="/chat" className="text-sm text-teal-600 hover:text-teal-700 font-medium transition-colors">
                Chat
              </Link>
              {['Features', 'API', 'Roadmap'].map(item => (
                <a key={item} href={`#${item.toLowerCase()}`}
                   className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                  {item}
                </a>
              ))}
            </nav>

            <div className="hidden md:flex items-center gap-2">
              <ThemeToggle />
              <Button variant="ghost" size="sm" asChild>
                <a href="https://github.com/Techgeek8000-byte/nexusai-website" target="_blank" rel="noopener noreferrer" className="gap-2">
                  <Github className="w-4 h-4" /> GitHub
                </a>
              </Button>
              <Button size="sm" className="bg-teal-600 hover:bg-teal-700 text-white gap-2" asChild>
                <Link href="/chat">Open Chat <ArrowRight className="w-4 h-4" /></Link>
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <div className="md:hidden flex items-center gap-1">
              <ThemeToggle />
              <button className="p-2" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
                {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
            </div>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden border-t border-border/50 py-4 space-y-3">
              {['Features', 'API', 'Roadmap'].map(item => (
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
          <div className="absolute inset-0 bg-gradient-to-b from-teal-500/5 to-transparent dark:from-teal-500/10" />
          <div className="absolute top-20 left-1/4 w-72 h-72 bg-teal-500/10 rounded-full blur-3xl" />
          <div className="absolute top-40 right-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl" />

          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-24 sm:pt-28 sm:pb-32">
            <div className="text-center max-w-4xl mx-auto">
              <Badge variant="secondary" className="mb-6 px-4 py-1.5 text-sm bg-teal-100 text-teal-800 border-teal-200 dark:bg-teal-900/40 dark:text-teal-300 dark:border-teal-800">
                100% Free &middot; No Signup Required
              </Badge>

              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight leading-tight">
                Free AI That{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-600 to-cyan-600">
                  Actually Works
                </span>
              </h1>

              <p className="mt-6 text-lg sm:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed">
                Multi-model chat, real-time streaming, code execution in 20+ languages, web search with sources, and an OpenAI-compatible API. All free, no account needed.
              </p>

              <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button size="lg" className="bg-teal-600 hover:bg-teal-700 text-white px-8 h-12 text-base gap-2 shadow-lg shadow-teal-600/20" asChild>
                  <Link href="/chat"><Sparkles className="w-5 h-5" /> Start Chatting Now</Link>
                </Button>
                <Button size="lg" variant="outline" className="px-8 h-12 text-base gap-2" asChild>
                  <a href="#api"><Code2 className="w-5 h-5" /> View API Docs</a>
                </Button>
              </div>

              {/* Trust Signals */}
              <div className="mt-10 flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-xs text-muted-foreground">
                <span className="flex items-center gap-1.5"><Check className="w-3.5 h-3.5 text-teal-600" /> No signup required</span>
                <span className="flex items-center gap-1.5"><Check className="w-3.5 h-3.5 text-teal-600" /> No API key needed</span>
                <span className="flex items-center gap-1.5"><Check className="w-3.5 h-3.5 text-teal-600" /> Open source</span>
                <span className="flex items-center gap-1.5"><Check className="w-3.5 h-3.5 text-teal-600" /> Data stays in your browser</span>
              </div>
            </div>

            {/* Stats Bar */}
            <div className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-6 max-w-3xl mx-auto">
              {STATS.map(stat => (
                <div key={stat.label} className="text-center p-4 rounded-xl bg-card border border-border/50">
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
              <Badge variant="secondary" className="mb-4">What's Included</Badge>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
                Everything You Need, Nothing You Don't
              </h2>
              <p className="mt-4 text-muted-foreground text-lg">
                Every feature listed below is live and working right now. No &quot;coming soon&quot;, no placeholders.
              </p>
            </div>

            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {FEATURES.map(feature => (
                <Card key={feature.title} className="group hover:shadow-lg hover:border-teal-200 dark:hover:border-teal-800 transition-all duration-300 border-border/50">
                  <CardHeader>
                    <div className="flex items-center justify-between mb-2">
                      <div className="w-12 h-12 rounded-xl bg-teal-50 dark:bg-teal-900/30 flex items-center justify-center text-teal-600 group-hover:bg-teal-100 dark:group-hover:bg-teal-900/50 transition-colors">
                        {feature.icon}
                      </div>
                      <Badge variant="outline" className={`text-xs ${feature.tag === 'Live' ? 'text-emerald-600 border-emerald-200 dark:border-emerald-800' : 'text-blue-600 border-blue-200 dark:border-blue-800'}`}>
                        {feature.tag}
                      </Badge>
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

            {/* Additional features row */}
            <div className="mt-8 grid sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-5xl mx-auto">
              {[
                { icon: <FileText className="w-4 h-4" />, text: 'File upload & analysis' },
                { icon: <Upload className="w-4 h-4" />, text: 'Export chats as Markdown' },
                { icon: <Cpu className="w-4 h-4" />, text: 'Auto model fallback (7B→3B→0.5B)' },
                { icon: <MessageSquare className="w-4 h-4" />, text: 'Conversation history (50 saved)' },
              ].map(f => (
                <div key={f.text} className="flex items-center gap-2.5 px-4 py-3 rounded-lg bg-card border border-border/50 text-sm text-muted-foreground">
                  <span className="text-teal-600">{f.icon}</span>
                  {f.text}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ━━━ Try It Now Section ━━━ */}
        <section className="py-20 sm:py-28 bg-gradient-to-b from-muted/30 to-background">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <Badge variant="secondary" className="mb-4">Try It Now</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
              Don't Take Our Word For It
            </h2>
            <p className="mt-4 text-muted-foreground text-lg mb-8">
              The chat is live. Ask anything, run code, search the web. No signup, no waiting.
            </p>

            {/* Interactive prompt suggestions */}
            <div className="grid sm:grid-cols-2 gap-3 max-w-2xl mx-auto">
              {SUGGESTED_PROMPTS.map((prompt, i) => (
                <Link
                  key={i}
                  href={`/chat?prompt=${encodeURIComponent(prompt)}`}
                  className="flex items-center gap-3 p-4 rounded-xl border border-border/50 bg-card hover:border-teal-200 dark:hover:border-teal-800 hover:shadow-md transition-all text-left group"
                >
                  <div className="w-8 h-8 rounded-lg bg-teal-50 dark:bg-teal-900/30 flex items-center justify-center text-teal-600 shrink-0 group-hover:bg-teal-100 dark:group-hover:bg-teal-900/50 transition-colors">
                    <Play className="w-4 h-4" />
                  </div>
                  <span className="text-sm text-muted-foreground group-hover:text-foreground transition-colors">{prompt}</span>
                </Link>
              ))}
            </div>

            <div className="mt-8">
              <Button size="lg" className="bg-teal-600 hover:bg-teal-700 text-white px-10 h-12 text-base gap-2 shadow-lg shadow-teal-600/20" asChild>
                <Link href="/chat"><Sparkles className="w-5 h-5" /> Open Full Chat</Link>
              </Button>
            </div>
          </div>
        </section>

        {/* ━━━ API Documentation Section ━━━ */}
        <section id="api" className="py-20 sm:py-28">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-2xl mx-auto mb-16">
              <Badge variant="secondary" className="mb-4">For Developers</Badge>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
                OpenAI-Compatible API
              </h2>
              <p className="mt-4 text-muted-foreground text-lg">
                Change one line in your existing code. That's it.
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
                      <Network className="w-5 h-5 text-teal-600" /> Available Endpoints
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {API_ENDPOINTS.map(ep => (
                        <div key={ep.path} className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
                          <code className={`text-xs font-bold px-2 py-1 rounded ${
                            ep.method === 'GET' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                          }`}>{ep.method}</code>
                          <code className="text-sm font-mono flex-1">{ep.path}</code>
                          <span className="text-xs text-muted-foreground hidden sm:block">{ep.desc}</span>
                        </div>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-4">
                      No API key required for the native endpoint. For the OpenAI-compatible endpoint, pass any string as the API key.
                    </p>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="quickstart">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Code2 className="w-5 h-5 text-teal-600" /> Python Quick Start
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="p-4 bg-zinc-900 dark:bg-zinc-950 rounded-lg overflow-x-auto">
                        <pre className="text-sm text-green-400 font-mono whitespace-pre">{`# pip install openai

from openai import OpenAI

client = OpenAI(
    base_url="https://nexusai-your-url.vercel.app/v1",
    api_key="any-value"  # not checked
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")`}</pre>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Badge variant="outline" className="text-xs"><Check className="w-3 h-3 mr-1" /> Drop-in replacement</Badge>
                        <Badge variant="outline" className="text-xs"><Check className="w-3 h-3 mr-1" /> Streaming support</Badge>
                        <Badge variant="outline" className="text-xs"><Check className="w-3 h-3 mr-1" /> 4 models available</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </section>

        {/* ━━━ Roadmap Section ━━━ */}
        <section id="roadmap" className="py-20 sm:py-28 bg-gradient-to-b from-muted/30 to-background">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-2xl mx-auto mb-16">
              <Badge variant="secondary" className="mb-4">Roadmap</Badge>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
                What's Next
              </h2>
              <p className="mt-4 text-muted-foreground text-lg">
                Built fast, shipping faster. Here's what's done and what's coming.
              </p>
            </div>

            <div className="max-w-3xl mx-auto space-y-6">
              {[
                { phase: 'Phase 1', title: 'Core Platform', timeline: 'Shipped', status: 'done', items: [
                  'Multi-model chat (4 Qwen models)', 'Real-time SSE streaming', 'Conversation history (localStorage)', 'OpenAI-compatible API', 'Model fallback chain (7B→3B→0.5B)', 'Dark/light theme', 'File upload & analysis', 'Chat export (Markdown)'
                ] },
                { phase: 'Phase 2', title: 'Agent Tools', timeline: 'Shipped', status: 'done', items: [
                  'Calculator (sanitized eval)', 'Date/time tool', 'Code execution (Piston API, 20+ langs)', 'Web search (Tavily + Wikipedia fallback)', 'Server-side function calling', 'Client-side quick detection'
                ] },
                { phase: 'Phase 3', title: 'Polish & Scale', timeline: 'In Progress', status: 'current', items: [
                  'User accounts (Supabase auth)', 'Cloud conversation sync', 'Conversation search', 'Keyboard shortcuts', 'Mobile PWA support'
                ] },
                { phase: 'Phase 4', title: 'Custom Models', timeline: 'Planned', status: 'upcoming', items: [
                  'Fine-tuned models (Google Colab)', 'Urdu-optimized model', 'Additional providers (Groq, OpenRouter)', 'Model evaluation benchmarks', 'Community model sharing'
                ] }
              ].map((phase, idx) => (
                <Card key={phase.phase} className="border-border/50">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold ${
                          phase.status === 'done' ? 'bg-teal-600 text-white' :
                          phase.status === 'current' ? 'bg-amber-500 text-white' :
                          'bg-muted text-muted-foreground'
                        }`}>
                          {idx + 1}
                        </div>
                        <div>
                          <CardTitle className="text-base">{phase.phase}: {phase.title}</CardTitle>
                          <CardDescription>{phase.timeline}</CardDescription>
                        </div>
                      </div>
                      {phase.status === 'done' && (
                        <Badge className="bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 border-0">Shipped</Badge>
                      )}
                      {phase.status === 'current' && (
                        <Badge className="bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 border-0">In Progress</Badge>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <ul className="grid sm:grid-cols-2 gap-2">
                      {phase.items.map(item => (
                        <li key={item} className="flex items-center gap-2 text-sm">
                          <Check className={`w-4 h-4 flex-shrink-0 ${phase.status === 'done' ? 'text-teal-600' : 'text-muted-foreground'}`} /> {item}
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
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center mx-auto mb-6">
                <Sparkles className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
                Ready to Try It?
              </h2>
              <p className="mt-4 text-lg text-muted-foreground">
                No signup. No API key. No credit card. Just open the chat and start.
              </p>
              <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
                <Button size="lg" className="bg-teal-600 hover:bg-teal-700 text-white px-10 h-12 text-base gap-2 shadow-lg shadow-teal-600/20" asChild>
                  <Link href="/chat"><Sparkles className="w-5 h-5" /> Open AI Chat</Link>
                </Button>
                <Button size="lg" variant="outline" className="px-8 h-12 text-base gap-2" asChild>
                  <a href="https://github.com/Techgeek8000-byte/nexusai-website" target="_blank" rel="noopener noreferrer">
                    <Github className="w-5 h-5" /> View Source
                  </a>
                </Button>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* ━━━ Footer ━━━ */}
      <footer className="border-t border-border/50 bg-muted/20 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div>
                <span className="text-lg font-bold">Nexus<span className="text-teal-600">AI</span></span>
                <p className="text-xs text-muted-foreground">Free AI Platform</p>
              </div>
            </div>
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <a href="#features" className="hover:text-foreground transition-colors">Features</a>
              <a href="#api" className="hover:text-foreground transition-colors">API</a>
              <a href="#roadmap" className="hover:text-foreground transition-colors">Roadmap</a>
              <a href="https://github.com/Techgeek8000-byte/nexusai-website" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors flex items-center gap-1"><Github className="w-3.5 h-3.5" /> Source</a>
            </div>
            <p className="text-xs text-muted-foreground">Built by Osama</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
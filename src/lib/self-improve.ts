'use client'

// ═══════════════════════════════════════════════════════════════
// NexusAI Self-Improvement Engine
// Learns from user feedback to optimize responses over time.
// All data stored in localStorage — no server needed.
// ═══════════════════════════════════════════════════════════════

// ── Types ──

export interface ResponseRating {
  id: string
  prompt: string
  response: string
  rating: 1 | -1
  category: string
  responseLength: number
  model: string
  timestamp: number
}

export interface LearningInsights {
  totalRated: number
  upCount: number
  downCount: number
  avgRating: number          // 0 to 1
  stylePreference: 'concise' | 'detailed' | 'balanced'
  urduAffinity: number       // 0 to 1
  topCategories: { category: string; score: number; count: number }[]
  totalSavedTokens: number   // estimated tokens saved via compression
}

export interface ContextResult {
  messages: any[]
  wasCompressed: boolean
  originalCount: number
  finalCount: number
}

// ── Storage ──

const RATINGS_KEY = 'nexusai_ratings'
const STATS_KEY = 'nexusai_improve_stats'
const MAX_RATINGS = 300
const COMPRESSION_THRESHOLD = 16

function loadRatings(): ResponseRating[] {
  try {
    const raw = localStorage.getItem(RATINGS_KEY)
    return raw ? JSON.parse(raw) : []
  } catch { return [] }
}

function saveRatings(ratings: ResponseRating[]): void {
  localStorage.setItem(RATINGS_KEY, JSON.stringify(ratings.slice(-MAX_RATINGS)))
}

function loadStats(): { totalSavedTokens: number } {
  try {
    const raw = localStorage.getItem(STATS_KEY)
    return raw ? JSON.parse(raw) : { totalSavedTokens: 0 }
  } catch { return { totalSavedTokens: 0 }
  }
}

function saveStats(stats: { totalSavedTokens: number }): void {
  localStorage.setItem(STATS_KEY, JSON.stringify(stats))
}

// ── Category Detection ──

const URDU_RANGE = /[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]/

export function detectCategory(text: string): string {
  const lower = text.toLowerCase()
  if (/```|def |func |class |import |const |let |fn |pub /i.test(text) ||
      /\b(python|javascript|typescript|rust|golang|java|c\+\+|html|css)\b/i.test(lower)) {
    return 'code'
  }
  if (/\b(calculate|compute|solve|\d+\s*[+\-*/]\s*\d|what is \d|how many)\b/i.test(lower)) {
    return 'math'
  }
  if (/\b(search|find|latest|news|current|recent|today|2024|2025|2026)\b/i.test(lower)) {
    return 'search'
  }
  if (URDU_RANGE.test(text)) {
    return 'urdu'
  }
  if (/\b(write|story|poem|essay|creative|imagine|draft|compose)\b/i.test(lower)) {
    return 'creative'
  }
  if (/\b(explain|what is|how does|why does|describe|define|tell me about)\b/i.test(lower)) {
    return 'explanation'
  }
  if (/\b(run|execute|test|debug|output|error|fix|compile)\b/i.test(lower)) {
    return 'code'
  }
  return 'general'
}

// ── Rate a Response ──

export function rateResponse(
  prompt: string,
  response: string,
  rating: 1 | -1,
  model: string
): void {
  const ratings = loadRatings()
  const category = detectCategory(prompt)

  // Update existing rating for same prompt (if user changes their mind)
  const existingIdx = ratings.findIndex(r => r.prompt === prompt && r.response === response)
  if (existingIdx >= 0) {
    ratings[existingIdx].rating = rating
    ratings[existingIdx].timestamp = Date.now()
  } else {
    ratings.push({
      id: `r_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
      prompt: prompt.slice(0, 200),
      response: response.slice(0, 500),
      rating,
      category,
      responseLength: response.length,
      model,
      timestamp: Date.now(),
    })
  }

  saveRatings(ratings)
}

// Get rating for a specific response (to show UI state)
export function getResponseRating(prompt: string, response: string): 1 | -1 | null {
  const ratings = loadRatings()
  const found = ratings.find(r => r.prompt === prompt && r.response === response)
  return found ? found.rating : null
}

// ── Learning Analysis ──

export function getInsights(): LearningInsights {
  const ratings = loadRatings()
  const stats = loadStats()

  if (ratings.length === 0) {
    return {
      totalRated: 0, upCount: 0, downCount: 0, avgRating: 0.5,
      stylePreference: 'balanced', urduAffinity: 0.5,
      topCategories: [], totalSavedTokens: stats.totalSavedTokens,
    }
  }

  const upRatings = ratings.filter(r => r.rating === 1)
  const downRatings = ratings.filter(r => r.rating === -1)

  // Style preference based on response length
  const avgUpLen = upRatings.length > 0
    ? upRatings.reduce((s, r) => s + r.responseLength, 0) / upRatings.length
    : 500
  const avgDownLen = downRatings.length > 0
    ? downRatings.reduce((s, r) => s + r.responseLength, 0) / downRatings.length
    : 500

  let stylePreference: 'concise' | 'detailed' | 'balanced' = 'balanced'
  if (upRatings.length >= 3) {
    const ratio = avgUpLen / Math.max(avgDownLen, 1)
    if (ratio > 1.3) stylePreference = 'detailed'
    else if (ratio < 0.7) stylePreference = 'concise'
  }

  // Urdu affinity
  const urduUp = upRatings.filter(r => r.category === 'urdu').length
  const urduTotal = ratings.filter(r => r.category === 'urdu').length
  const urduAffinity = urduTotal > 0 ? urduUp / urduTotal : 0.5

  // Category scores
  const catMap: Record<string, { up: number; down: number }> = {}
  for (const r of ratings) {
    if (!catMap[r.category]) catMap[r.category] = { up: 0, down: 0 }
    if (r.rating === 1) catMap[r.category].up++
    else catMap[r.category].down++
  }
  const topCategories = Object.entries(catMap)
    .map(([category, { up, down }]) => ({
      category,
      score: (up + 1) / (up + down + 2), // Bayesian average
      count: up + down,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)

  return {
    totalRated: ratings.length,
    upCount: upRatings.length,
    downCount: downRatings.length,
    avgRating: upRatings.length / ratings.length,
    stylePreference,
    urduAffinity,
    topCategories,
    totalSavedTokens: stats.totalSavedTokens,
  }
}

// ── Adaptive System Prompt ──

export function getAdaptivePrompt(basePrompt: string): string {
  const insights = getInsights()
  if (insights.totalRated < 3) return basePrompt // Need minimum data

  const additions: string[] = []

  // Style adaptation
  if (insights.stylePreference === 'detailed') {
    additions.push('The user prefers detailed, thorough responses with explanations and examples.')
  } else if (insights.stylePreference === 'concise') {
    additions.push('The user prefers concise, direct answers. Get to the point quickly.')
  }

  // Urdu affinity
  if (insights.urduAffinity > 0.7 && insights.urduAffinity !== 0.5) {
    additions.push('The user often communicates in Urdu. Respond in Urdu when they write in Urdu, but use English for technical content.')
  }

  // Category-specific tips based on what gets rated well
  const strongCats = insights.topCategories.filter(c => c.score > 0.7 && c.count >= 2)
  for (const cat of strongCats) {
    switch (cat.category) {
      case 'code':
        additions.push('For code questions: always include working code with comments, explain the approach, and mention edge cases.')
        break
      case 'explanation':
        additions.push('For explanations: use analogies, break into steps, and include examples.')
        break
      case 'math':
        additions.push('For math: show your work step by step, use the calculator tool, and verify the answer.')
        break
      case 'creative':
        additions.push('For creative writing: be vivid, use varied sentence structure, and show don\'t tell.')
        break
    }
  }

  // Weak categories - what to improve
  const weakCats = insights.topCategories.filter(c => c.score < 0.4 && c.count >= 2)
  for (const cat of weakCats) {
    if (cat.category === 'search') {
      additions.push('For search questions: always use the web_search tool and cite your sources.')
    }
  }

  if (additions.length === 0) return basePrompt

  return basePrompt + '\n\n[Adapted from ' + insights.totalRated + ' feedback signals]\n' + additions.map(a => '- ' + a).join('\n')
}

// ── Context Compression ──
// Reduces token usage by summarizing old messages while keeping recent ones intact.
// This lets the AI maintain context from long conversations without hitting token limits.

export function compressContext(messages: any[]): ContextResult {
  if (messages.length <= COMPRESSION_THRESHOLD) {
    return { messages, wasCompressed: false, originalCount: messages.length, finalCount: messages.length }
  }

  const systemMsg = messages[0]
  const conversationMsgs = messages.slice(1)

  // Keep the most recent 60% of messages in full
  const keepCount = Math.max(8, Math.floor(conversationMsgs.length * 0.6))
  const oldMsgs = conversationMsgs.slice(0, -keepCount)
  const recentMsgs = conversationMsgs.slice(-keepCount)

  // Build compressed summary from old messages
  const summaryLines: string[] = []
  let tokenSavings = 0

  for (const msg of oldMsgs) {
    const role = msg.role === 'user' ? 'User' : 'AI'
    const originalLen = msg.content.length
    const isCode = msg.content.includes('```')
    const maxLen = isCode ? 120 : 150

    let excerpt = msg.content.replace(/\n{3,}/g, '\n\n').trim()
    if (excerpt.length > maxLen) {
      excerpt = excerpt.slice(0, maxLen).trim() + '...'
    }
    tokenSavings += Math.max(0, originalLen - excerpt.length)

    summaryLines.push(`${role}: ${excerpt}`)
  }

  const compressedSummary = {
    role: 'system' as const,
    content: (
      `[Compressed context: ${oldMsgs.length} earlier messages summarized for efficiency. ` +
      `Recent ${recentMsgs.length} messages below are in full.]\n\n` +
      summaryLines.join('\n\n')
    ),
  }

  // Track saved tokens
  const stats = loadStats()
  stats.totalSavedTokens += Math.floor(tokenSavings / 4) // rough token estimate
  saveStats(stats)

  return {
    messages: [systemMsg, compressedSummary, ...recentMsgs],
    wasCompressed: true,
    originalCount: messages.length,
    finalCount: 2 + recentMsgs.length,
  }
}

// ── Find Similar Past Responses ──
// If the user asks something similar to a highly-rated past question, reference it.

export function findRelevantPast(query: string): { prompt: string; response: string; rating: 1 | -1 } | null {
  const ratings = loadRatings()
  const queryLower = query.toLowerCase().slice(0, 100)
  const queryWords = new Set(queryLower.split(/\s+/).filter(w => w.length > 3))

  if (queryWords.size < 2) return null

  let bestMatch: ResponseRating | null = null
  let bestScore = 0

  for (const r of ratings) {
    if (r.rating !== 1) continue // only reference highly-rated responses
    const rWords = new Set(r.prompt.toLowerCase().split(/\s+/).filter(w => w.length > 3))
    let overlap = 0
    for (const w of queryWords) {
      if (rWords.has(w)) overlap++
    }
    const score = overlap / Math.min(queryWords.size, rWords.size)
    if (score > 0.5 && score > bestScore) {
      bestScore = score
      bestMatch = r
    }
  }

  if (!bestMatch) return null

  return {
    prompt: bestMatch.prompt,
    response: bestMatch.response,
    rating: bestMatch.rating,
  }
}

// ── Reset ──

export function resetLearning(): void {
  localStorage.removeItem(RATINGS_KEY)
  localStorage.removeItem(STATS_KEY)
}

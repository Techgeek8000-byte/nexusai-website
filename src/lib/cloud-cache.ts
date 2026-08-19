// ═══════════════════════════════════════════════════════════════
// NexusAI Cloud Response Cache
// Stores self-improved responses in Vercel KV (Redis) so all users
// benefit from past improvements. Falls back gracefully if KV is
// not configured — self-analysis still works, just no cross-session cache.
// ═══════════════════════════════════════════════════════════════

// ── Types ──

export interface CachedResponse {
  /** The improved response text */
  response: string;
  /** Quality score from self-analysis (0-100) */
  qualityScore: number;
  /** Issues that were found and fixed */
  issuesFixed: string[];
  /** Which model generated this */
  model: string;
  /** Timestamp when this was cached */
  cachedAt: number;
  /** How many times this cached response has been served */
  hitCount: number;
  /** Category of the query */
  category: string;
}

export interface CacheLookupResult {
  found: boolean;
  response?: string;
  qualityScore?: number;
  wasImproved?: boolean;
  cacheHit: boolean;
}

// ── Configuration ──

const CACHE_PREFIX = 'nexusai:resp:';
const CACHE_TTL_SECONDS = 7 * 24 * 60 * 60; // 7 days
const MAX_CACHE_SIZE = 500;
const INDEX_KEY = 'nexusai:cache_index';
const STATS_KEY = 'nexusai:cache_stats';

// ── Lazy-load Vercel KV ──
// We dynamically import so it doesn't crash if @vercel/kv isn't installed

let kvModule: any = null;
let kvAvailable: boolean | null = null;
let kvChecked = false;

async function getKV(): Promise<any | null> {
  if (kvChecked) return kvAvailable ? kvModule : null;
  kvChecked = true;

  // Check if KV env vars exist
  if (!process.env.KV_REST_URL || !process.env.KV_REST_TOKEN) {
    kvAvailable = false;
    return null;
  }

  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    kvModule = await import('@vercel/kv' as string).then((m: any) => m.kv || m);
    kvAvailable = !!kvModule;
    if (!kvAvailable) console.warn('[NexusAI Cache] @vercel/kv imported but kv export not found');
    return kvAvailable ? kvModule : null;
  } catch {
    kvAvailable = false;
    return null;
  }
}

// ── Query Normalization (for cache key matching) ──

export function normalizeQuery(query: string): string {
  return query
    .toLowerCase()
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/[!?.,;:]+$/g, '')
    .slice(0, 200);
}

function hashString(str: string): string {
  // Simple but fast hash for cache keys
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit int
  }
  return Math.abs(hash).toString(36);
}

function getCacheKey(query: string): string {
  return CACHE_PREFIX + hashString(normalizeQuery(query));
}

// ── Category Detection (lightweight, server-side) ──

function detectCategory(query: string): string {
  const lower = query.toLowerCase();
  if (/```|\b(def|func|class|import|const|let|fn |pub )\b/i.test(lower) ||
      /\b(python|javascript|typescript|rust|golang|java|c\++|html|css)\b/i.test(lower)) return 'code';
  if (/\b(calculate|compute|solve|\d+\s*[+\-*/]\s*\d)\b/i.test(lower)) return 'math';
  if (/\b(search|find|latest|news|current|recent)\b/i.test(lower)) return 'search';
  return 'general';
}

// ── Store Improved Response ──

export async function cacheImprovedResponse(
  query: string,
  improvedResponse: string,
  qualityScore: number,
  model: string,
  issuesFixed: string[]
): Promise<void> {
  const kv = await getKV();
  if (!kv) return; // Graceful: no KV = no caching, self-analysis still worked

  const key = getCacheKey(query);
  const cached: CachedResponse = {
    response: improvedResponse.slice(0, 8000),
    qualityScore,
    issuesFixed,
    model,
    cachedAt: Date.now(),
    hitCount: 0,
    category: detectCategory(query),
  };

  try {
    // Store the response
    await kv.set(key, JSON.stringify(cached), { ex: CACHE_TTL_SECONDS });

    // Add to index for management
    await kv.sadd(INDEX_KEY, key);

    // Update stats
    await kv.incr(STATS_KEY + ':total_improvements');
    await kv.incr(STATS_KEY + ':issues_fixed', issuesFixed.length);

    // Prune if too many entries
    await pruneIfNeeded(kv);
  } catch (err) {
    // KV errors should never break the chat
    console.warn('[NexusAI Cache] Write error:', err);
  }
}

// ── Lookup Cached Response ──

export async function lookupCachedResponse(query: string): Promise<CacheLookupResult> {
  const kv = await getKV();
  if (!kv) return { found: false, cacheHit: false };

  const key = getCacheKey(query);

  try {
    const raw = await kv.get(key);
    if (!raw) return { found: false, cacheHit: false };

    const cached: CachedResponse = JSON.parse(raw as string);

    // Check TTL — if older than 7 days, consider stale
    const age = Date.now() - cached.cachedAt;
    if (age > CACHE_TTL_SECONDS * 1000) {
      await kv.del(key);
      return { found: false, cacheHit: false };
    }

    // Only return cached responses with decent quality
    if (cached.qualityScore < 60) {
      return { found: false, cacheHit: false };
    }

    // Increment hit count
    cached.hitCount++;
    await kv.set(key, JSON.stringify(cached), { ex: CACHE_TTL_SECONDS });
    await kv.incr(STATS_KEY + ':cache_hits');

    return {
      found: true,
      response: cached.response,
      qualityScore: cached.qualityScore,
      wasImproved: cached.issuesFixed.length > 0,
      cacheHit: true,
    };
  } catch (err) {
    console.warn('[NexusAI Cache] Read error:', err);
    return { found: false, cacheHit: false };
  }
}

// ── Discard Weak Responses ──
// Removes low-quality cached entries, keeping only the best

export async function discardWeakResponses(): Promise<number> {
  const kv = await getKV();
  if (!kv) return 0;

  try {
    const keys = await kv.smembers(INDEX_KEY);
    if (!keys || keys.length === 0) return 0;

    let discarded = 0;
    const keysToKeep: string[] = [];

    for (const key of keys) {
      try {
        const raw = await kv.get(key as string);
        if (!raw) continue;

        const cached: CachedResponse = JSON.parse(raw as string);

        // Discard if: low quality, old and never hit, or too many issues
        const age = Date.now() - cached.cachedAt;
        const shouldDiscard =
          cached.qualityScore < 50 ||
          (age > 3 * 24 * 60 * 60 * 1000 && cached.hitCount === 0) || // 3+ days old, never used
          cached.issuesFixed.length > 5; // Was very broken

        if (shouldDiscard) {
          await kv.del(key as string);
          discarded++;
        } else {
          keysToKeep.push(key as string);
        }
      } catch {
        continue;
      }
    }

    // Update index
    if (discarded > 0) {
      await kv.del(INDEX_KEY);
      if (keysToKeep.length > 0) {
        await kv.sadd(INDEX_KEY, ...keysToKeep);
      }
      await kv.incr(STATS_KEY + ':discarded', discarded);
    }

    return discarded;
  } catch (err) {
    console.warn('[NexusAI Cache] Cleanup error:', err);
    return 0;
  }
}

// ── Get Cache Stats ──

export async function getCacheStats(): Promise<{
  totalImprovements: number;
  totalIssuesFixed: number;
  cacheHits: number;
  cacheSize: number;
} | null> {
  const kv = await getKV();
  if (!kv) return null;

  try {
    const [improvements, issues, hits, indexKeys] = await Promise.all([
      kv.get(STATS_KEY + ':total_improvements'),
      kv.get(STATS_KEY + ':issues_fixed'),
      kv.get(STATS_KEY + ':cache_hits'),
      kv.scard(INDEX_KEY),
    ]);

    return {
      totalImprovements: Number(improvements || 0),
      totalIssuesFixed: Number(issues || 0),
      cacheHits: Number(hits || 0),
      cacheSize: Number(indexKeys || 0),
    };
  } catch {
    return null;
  }
}

// ── Internal: Prune if cache is too large ──

async function pruneIfNeeded(kv: any): Promise<void> {
  try {
    const size = await kv.scard(INDEX_KEY);
    if (size <= MAX_CACHE_SIZE) return;

    // Remove oldest 20% of entries
    const keys = await kv.smembers(INDEX_KEY);
    if (!keys || keys.length === 0) return;

    interface KeyWithAge {
      key: string;
      age: number;
    }
    const entries: KeyWithAge[] = [];

    for (const key of keys) {
      try {
        const raw = await kv.get(key as string);
        if (raw) {
          const cached: CachedResponse = JSON.parse(raw as string);
          entries.push({ key: key as string, age: cached.cachedAt });
        }
      } catch {
        continue;
      }
    }

    // Sort by age (oldest first)
    entries.sort((a, b) => a.age - b.age);

    // Remove oldest 20%
    const toRemove = Math.floor(entries.length * 0.2);
    for (let i = 0; i < toRemove; i++) {
      await kv.del(entries[i].key);
    }
  } catch {
    // Pruning failure is not critical
  }
}

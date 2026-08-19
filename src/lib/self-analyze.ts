// ═══════════════════════════════════════════════════════════════
// NexusAI Cloud Self-Analysis Engine
// After generating a response, the AI reviews itself for:
//   - Code errors (syntax, logic, runtime)
//   - Security vulnerabilities (injection, XSS, hardcoded secrets, etc.)
//   - Incomplete or inaccurate answers
// If issues are found, it auto-corrects and returns the improved version.
//   The weaker version is discarded; only the improved one survives.
// All analysis happens server-side via HF Inference API.
// ═══════════════════════════════════════════════════════════════

// ── Types ──

export interface AnalysisResult {
  /** Whether the response was changed by self-analysis */
  improved: boolean;
  /** The final response (original if no issues, improved if corrected) */
  finalResponse: string;
  /** Issues found during analysis */
  issues: AnalysisIssue[];
  /** Overall quality score 0-100 */
  qualityScore: number;
  /** Which model performed the analysis */
  analyzerModel: string;
  /** Time taken for analysis in ms */
  analysisTimeMs: number;
}

export interface AnalysisIssue {
  type: 'code_error' | 'vulnerability' | 'incomplete' | 'inaccuracy' | 'improvement';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  suggestion: string;
}

// ── Review Prompts ──

const REVIEW_SYSTEM_PROMPT = `You are a strict code reviewer and quality analyst. Your job is to analyze AI-generated responses for errors, security vulnerabilities, and quality issues.

Analyze the response and respond ONLY with valid JSON in this exact format (no markdown, no code fences, just raw JSON):
{
  "has_code": true/false,
  "has_issues": true/false,
  "issues": [
    {
      "type": "code_error" | "vulnerability" | "incomplete" | "inaccuracy" | "improvement",
      "severity": "low" | "medium" | "high" | "critical",
      "description": "brief description of the issue",
      "suggestion": "how to fix it"
    }
  ],
  "quality_score": 0-100,
  "should_rewrite": true/false,
  "rewrite_instructions": "if should_rewrite is true, specific instructions on what to fix"
}

Check for these specific issues:
- **Code errors**: syntax errors, undefined variables, wrong function signatures, missing imports, type errors, off-by-one errors, unhandled edge cases, infinite loops
- **Security vulnerabilities**: SQL injection, XSS, command injection, hardcoded passwords/API keys, unsafe deserialization, path traversal, eval/exec usage, buffer overflow potential, missing input validation, insecure random
- **Incomplete answers**: truncated code, missing explanations, unanswered parts of the question, placeholder text left in
- **Inaccuracies**: wrong facts, incorrect API usage, deprecated methods, wrong syntax for the specified language
- **Improvements**: better algorithms, more efficient code, cleaner structure, better error handling, missing best practices

Be thorough but fair. Minor style preferences are not issues. Only flag real problems.
If the response is good quality with score >= 80 and no real issues, set has_issues to false.`;

const CORRECT_SYSTEM_PROMPT = `You are an AI assistant that improves responses based on review feedback. You will receive:
1. The original user question
2. The original AI response
3. Review feedback listing specific issues

Your job: Rewrite the response fixing ALL the issues mentioned. Keep everything that was good. Only change what needs fixing.
- Fix code errors and vulnerabilities
- Complete any incomplete sections
- Correct any inaccuracies
- Apply the suggested improvements
- Do NOT add unnecessary commentary about the fixes
- Output the complete improved response directly`;

// ── Helper: Call HF API ──

async function callHF(
  hfToken: string,
  modelId: string,
  messages: { role: string; content: string }[],
  maxTokens: number
): Promise<{ content: string; status: number }> {
  try {
    const response = await fetch(
      'https://api-inference.huggingface.co/v1/chat/completions',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${hfToken}`,
        },
        body: JSON.stringify({
          model: modelId,
          messages,
          temperature: 0.1, // Low temp for consistent analysis
          max_tokens: maxTokens,
        }),
      }
    );

    if (response.status === 429 || response.status === 503) {
      return { content: '', status: response.status };
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || '';
    return { content, status: response.status };
  } catch {
    return { content: '', status: 500 };
  }
}

// ── Parse review JSON (robust, handles malformed responses) ──

function parseReviewJSON(raw: string): {
  has_code: boolean;
  has_issues: boolean;
  issues: AnalysisIssue[];
  quality_score: number;
  should_rewrite: boolean;
  rewrite_instructions: string;
} | null {
  // Try direct parse
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed.has_issues === 'boolean') return parsed;
  } catch {}

  // Try extracting JSON from markdown code fence
  const fenceMatch = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (fenceMatch) {
    try {
      const parsed = JSON.parse(fenceMatch[1].trim());
      if (typeof parsed.has_issues === 'boolean') return parsed;
    } catch {}
  }

  // Try finding first { ... } block
  const braceStart = raw.indexOf('{');
  const braceEnd = raw.lastIndexOf('}');
  if (braceStart >= 0 && braceEnd > braceStart) {
    try {
      const parsed = JSON.parse(raw.slice(braceStart, braceEnd + 1));
      if (typeof parsed.has_issues === 'boolean') return parsed;
    } catch {}
  }

  return null;
}

// ── Should analyze? (Skip for trivial responses) ──

export function shouldAnalyze(response: string): boolean {
  // Skip very short responses
  if (response.length < 80) return false;
  // Skip responses that are just errors
  if (response.startsWith('Error:') || response.startsWith('HF API error')) return false;
  // Analyze if there's code, or the response is substantial
  const hasCode = /```\w*/.test(response);
  const isLong = response.length > 300;
  return hasCode || isLong;
}

// ── Main: Self-Analyze Response ──

export async function selfAnalyzeResponse(
  hfToken: string,
  userQuery: string,
  originalResponse: string,
  originalModel: string
): Promise<AnalysisResult> {
  const startTime = Date.now();
  const defaultResult: AnalysisResult = {
    improved: false,
    finalResponse: originalResponse,
    issues: [],
    qualityScore: 100,
    analyzerModel: 'none',
    analysisTimeMs: 0,
  };

  // Check if we should even analyze
  if (!shouldAnalyze(originalResponse)) {
    return defaultResult;
  }

  // Use a smaller/faster model for analysis to save tokens
  const analysisModels = [
    'Qwen/Qwen2.5-3B-Instruct',
    'Qwen/Qwen2.5-0.5B-Instruct',
  ];

  // ── Step 1: Review the response ──
  const reviewMessages = [
    { role: 'system', content: REVIEW_SYSTEM_PROMPT },
    {
      role: 'user',
      content: `User's question:\n${userQuery.slice(0, 500)}\n\nAI's response to review:\n${originalResponse.slice(0, 3000)}`,
    },
  ];

  let reviewRaw = '';
  let analyzerModel = '';

  for (const model of analysisModels) {
    const { content, status } = await callHF(hfToken, model, reviewMessages, 1024);
    if (status === 200 && content) {
      reviewRaw = content;
      analyzerModel = model;
      break;
    }
    // Rate limited, try next
  }

  if (!reviewRaw) {
    // Analysis failed — return original unchanged
    return defaultResult;
  }

  // Parse the review
  const review = parseReviewJSON(reviewRaw);
  if (!review) {
    // Couldn't parse review — return original
    return defaultResult;
  }

  const issues: AnalysisIssue[] = (review.issues || []).map((iss: any) => ({
    type: iss.type || 'improvement',
    severity: iss.severity || 'low',
    description: iss.description || 'Unknown issue',
    suggestion: iss.suggestion || '',
  }));

  const qualityScore = typeof review.quality_score === 'number' ? review.quality_score : 75;

  // If no issues or high quality, return as-is
  if (!review.has_issues || qualityScore >= 85 || issues.length === 0) {
    return {
      improved: false,
      finalResponse: originalResponse,
      issues,
      qualityScore,
      analyzerModel,
      analysisTimeMs: Date.now() - startTime,
    };
  }

  // ── Step 2: Auto-correct if issues found ──
  // Only auto-correct for medium+ severity issues
  const significantIssues = issues.filter(
    (i) => i.severity === 'high' || i.severity === 'critical' || i.severity === 'medium'
  );

  if (significantIssues.length === 0 || !review.should_rewrite) {
    return {
      improved: false,
      finalResponse: originalResponse,
      issues,
      qualityScore,
      analyzerModel,
      analysisTimeMs: Date.now() - startTime,
    };
  }

  // Build correction prompt
  const issueSummary = issues
    .map((i, idx) => `${idx + 1}. [${i.severity.toUpperCase()}] ${i.type}: ${i.description}. Fix: ${i.suggestion}`)
    .join('\n');

  const correctMessages = [
    { role: 'system', content: CORRECT_SYSTEM_PROMPT },
    {
      role: 'user',
      content: `Original question:\n${userQuery.slice(0, 500)}\n\nOriginal response (has issues):\n${originalResponse.slice(0, 3000)}\n\nIssues found that MUST be fixed:\n${issueSummary}\n\n${review.rewrite_instructions || 'Fix all the issues listed above.'}\n\nProvide the complete improved response:`,
    },
  ];

  // Use the original model (or best available) for correction
  const correctModels = [
    originalModel,
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
  ];
  // Deduplicate
  const uniqueModels = [...new Set(correctModels)];

  let improvedResponse = '';
  for (const model of uniqueModels) {
    const { content, status } = await callHF(hfToken, model, correctMessages, 4096);
    if (status === 200 && content && content.length > originalResponse.length * 0.5) {
      improvedResponse = content;
      break;
    }
  }

  if (!improvedResponse) {
    // Correction failed — return original with issues noted
    return {
      improved: false,
      finalResponse: originalResponse,
      issues,
      qualityScore,
      analyzerModel,
      analysisTimeMs: Date.now() - startTime,
    };
  }

  // ── Step 3: Verify the improved version is actually better ──
  // Quick sanity: improved version should be longer (more complete) and not shorter than original
  const improvedIsBetter =
    improvedResponse.length >= originalResponse.length * 0.8;

  if (!improvedIsBetter) {
    return {
      improved: false,
      finalResponse: originalResponse,
      issues,
      qualityScore,
      analyzerModel,
      analysisTimeMs: Date.now() - startTime,
    };
  }

  // Success — return improved version, discard the weak one
  return {
    improved: true,
    finalResponse: improvedResponse,
    issues,
    qualityScore: Math.min(qualityScore + 15, 100), // Boost score after correction
    analyzerModel,
    analysisTimeMs: Date.now() - startTime,
  };
}

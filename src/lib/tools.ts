// ━━━ Tool Definitions (sent to model for function calling) ━━━
export const TOOL_DEFINITIONS = [
  {
    type: "function" as const,
    function: {
      name: "calculator",
      description:
        "Evaluate a mathematical expression. Supports +, -, *, /, parentheses, decimals, and powers (**).",
      parameters: {
        type: "object",
        properties: {
          expression: {
            type: "string",
            description:
              'The mathematical expression to evaluate, e.g. "(25 * 14) + (376 / 4)"',
          },
        },
        required: ["expression"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "get_current_datetime",
      description:
        "Get the current date, time, and day of the week. Use when the user asks about the current time or date.",
      parameters: {
        type: "object",
        properties: {},
        required: [],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "execute_code",
      description:
        "Execute code in a sandboxed environment. Supports Python, JavaScript, TypeScript, Go, Rust, C, C++, Java, Ruby, and more. Returns stdout and stderr. Use this when the user asks to run, execute, or test code.",
      parameters: {
        type: "object",
        properties: {
          language: {
            type: "string",
            description:
              'Programming language. Common values: "python", "javascript", "typescript", "go", "rust", "c", "cpp", "java", "ruby", "bash"',
          },
          code: {
            type: "string",
            description: "The source code to execute.",
          },
        },
        required: ["language", "code"],
      },
    },
  },
  {
    type: "function" as const,
    function: {
      name: "web_search",
      description:
        "Search the web for current information. Use when the user asks about recent events, news, current data, or anything that requires up-to-date information.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The search query.",
          },
        },
        required: ["query"],
      },
    },
  },
];

export interface ToolCall {
  name: string;
  arguments: Record<string, any>;
}

// ━━━ Tool Execution ━━━
export async function executeTool(call: ToolCall): Promise<string> {
  switch (call.name) {
    case "calculator":
      return executeCalculator(call.arguments.expression);
    case "get_current_datetime":
      return getDatetime();
    case "execute_code":
      return executeCode(call.arguments.language, call.arguments.code);
    case "web_search":
      return webSearch(call.arguments.query);
    default:
      return `Unknown tool: ${call.name}`;
  }
}

function executeCalculator(expression: string): string {
  const sanitized = expression.trim();
  const allowed = new Set("0123456789+-*/.() ");
  if (![...sanitized].every((c) => allowed.has(c))) {
    return `Error: Invalid characters in expression "${expression}". Only numbers, +, -, *, /, (, ) are allowed.`;
  }
  if (!/\d/.test(sanitized)) {
    return "Error: Expression must contain at least one number.";
  }
  try {
    const fn = new Function(`"use strict"; return (${sanitized})`)();
    if (typeof fn !== "number" || !isFinite(fn)) {
      return "Error: Invalid expression or division by zero.";
    }
    const result = Math.round(fn * 1e10) / 1e10;
    return `${sanitized} = ${result}`;
  } catch {
    return `Error: Could not evaluate expression "${expression}". Check syntax.`;
  }
}

function getDatetime(): string {
  return new Date().toLocaleString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
  });
}

// ━━━ Code Execution via Piston API (free, sandboxed) ━━━
const PISTON_API = "https://emkc.org/api/v2/piston/execute";

const LANGUAGE_MAP: Record<string, string> = {
  python: "python3",
  javascript: "javascript",
  js: "javascript",
  typescript: "typescript",
  ts: "typescript",
  go: "go",
  rust: "rust",
  c: "c",
  cpp: "c++",
  "c++": "c++",
  java: "java",
  ruby: "ruby",
  rb: "ruby",
  bash: "bash",
  sh: "bash",
  shell: "bash",
  swift: "swift",
  kotlin: "kotlin",
  r: "r",
  lua: "lua",
  php: "php",
  sql: "sql",
};

async function executeCode(
  language: string,
  code: string
): Promise<string> {
  const lang = LANGUAGE_MAP[language.toLowerCase()] || language.toLowerCase();

  try {
    const response = await fetch(PISTON_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        language: lang,
        version: "*",
        files: [{ content: code }],
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      return `Code execution error (HTTP ${response.status}): ${errText}`;
    }

    const data = await response.json();

    // Piston returns { run: { stdout, stderr, exitCode, signal, output } }
    const run = data.run;
    if (!run) return "Code execution returned no output.";

    const parts: string[] = [];
    if (run.stdout) parts.push(run.stdout.trim());
    if (run.stderr) parts.push(`[stderr]\n${run.stderr.trim()}`);
    if (run.exitCode !== 0)
      parts.push(`[Process exited with code ${run.exitCode}]`);

    return parts.length > 0 ? parts.join("\n") : "(no output)";
  } catch (error) {
    return `Code execution failed: ${error instanceof Error ? error.message : String(error)}`;
  }
}

// ━━━ Web Search ━━━
async function webSearch(query: string): Promise<string> {
  // Try Tavily first (if API key is set)
  const tavilyKey = process.env.TAVILY_API_KEY;
  if (tavilyKey) {
    try {
      const res = await fetch("https://api.tavily.com/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          api_key: tavilyKey,
          max_results: 3,
          include_answer: true,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.answer) {
          let result = `Search results for "${query}":\n\n${data.answer}`;
          if (data.results?.length > 0) {
            result += "\n\nSources:\n";
            data.results.forEach(
          (r: { title: string; url: string }, i: number) => {
            result += `${i + 1}. ${r.title}: ${r.url}\n`;
          }
        );
          }
          return result;
        }
      }
    } catch {
      // Fall through to Wikipedia
    }
  }

  // Fallback: Wikipedia API (free, no key needed)
  try {
    const res = await fetch(
      `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(
        query
      )}`
    );
    if (res.ok) {
      const data = await res.json();
      if (data.extract) {
        return `From Wikipedia - "${data.title}":\n${data.extract}\n\nSource: ${data.content_urls?.desktop?.page || ""}`;
      }
    }
    return `No Wikipedia results found for "${query}". For full web search, set TAVILY_API_KEY in environment variables (free at tavily.com).`;
  } catch {
    return `Search failed for "${query}". Web search requires TAVILY_API_KEY environment variable.`;
  }
}

// ━━━ Client-side quick tool detection (regex, for instant results) ━━━
export function detectToolsClient(
  message: string
): { name: string; result: string }[] {
  const results: { name: string; result: string }[] = [];

  // Calculator
  const calcMatch = message.match(
    /(?:calculate|compute|solve|what(?:'s| is))\s+([\d+\-*/().\s]{3,})/i
  );
  if (calcMatch) {
    const expr = calcMatch[1].trim();
    if (/\d/.test(expr) && /[+\-*/]/.test(expr)) {
      try {
        const allowed = new Set("0123456789+-*/.() ");
        if ([...expr].every((c) => allowed.has(c))) {
          const val = new Function(`"use strict"; return (${expr})`)();
          results.push({
            name: "Calculator",
            result: `${expr} = ${val}`,
          });
        }
      } catch {
        /* skip */
      }
    }
  }

  // Date/Time
  if (
    /(?:what(?:'s| is) the (?:current )?(?:date|time|day)|what time|current (?:date|time)|tariekh|waqt)/i.test(
      message
    )
  ) {
    results.push({ name: "Date/Time", result: getDatetime() });
  }

  return results;
}

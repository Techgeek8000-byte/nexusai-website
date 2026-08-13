// ━━━ Server-Side Agent Tools ━━━
// These run on the server (API route) — not client-side string injection.

interface ToolCall {
  name: string
  arguments: Record<string, any>
}

interface ToolResult {
  name: string
  result: string
}

const TOOL_DEFINITIONS = [
  {
    type: 'function' as const,
    function: {
      name: 'calculator',
      description: 'Evaluate a mathematical expression and return the result. Supports +, -, *, /, parentheses, and decimal numbers.',
      parameters: {
        type: 'object',
        properties: {
          expression: { type: 'string', description: 'The mathematical expression to evaluate, e.g. "(25 * 14) + (376 / 4)"' },
        },
        required: ['expression'],
      },
    },
  },
  {
    type: 'function' as const,
    function: {
      name: 'get_current_datetime',
      description: 'Get the current date and time. Useful when the user asks about the current time, date, or day of the week.',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
]

export { TOOL_DEFINITIONS }
export type { ToolCall, ToolResult }

// ━━━ Tool Execution ━━━
export function executeTool(call: ToolCall): string {
  switch (call.name) {
    case 'calculator':
      return executeCalculator(call.arguments.expression)
    case 'get_current_datetime':
      return new Date().toLocaleString('en-US', {
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZoneName: 'short',
      })
    default:
      return `Unknown tool: ${call.name}`
  }
}

function executeCalculator(expression: string): string {
  // Sanitize: only allow digits, operators, parentheses, decimal points, spaces
  const sanitized = expression.trim()
  const allowed = new Set('0123456789+-*/.() ')
  if (![...sanitized].every(c => allowed.has(c))) {
    return `Error: Invalid characters in expression "${expression}". Only numbers and +, -, *, /, (, ) are allowed.`
  }
  if (!/\d/.test(sanitized)) {
    return 'Error: Expression must contain at least one number.'
  }
  try {
    // Use Function constructor for safe eval (no access to scope)
    const fn = new Function(`"use strict"; return (${sanitized})`)()
    if (typeof fn !== 'number' || !isFinite(fn)) {
      return `Error: Invalid expression or division by zero.`
    }
    // Round to avoid floating point display issues
    const result = Math.round(fn * 1e10) / 1e10
    return `${sanitized} = ${result}`
  } catch {
    return `Error: Could not evaluate expression "${expression}". Check your syntax.`
  }
}

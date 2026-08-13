import { NextRequest, NextResponse } from "next/server";
import { TOOL_DEFINITIONS, executeTool } from "@/lib/tools";

const MODEL_MAP: Record<string, string> = {
  "Qwen 2.5 7B (Best Quality)": "Qwen/Qwen2.5-7B-Instruct",
  "Qwen 2.5 3B (Balanced)": "Qwen/Qwen2.5-3B-Instruct",
  "Qwen 2.5 0.5B (Fastest)": "Qwen/Qwen2.5-0.5B-Instruct",
  "CodeQwen 7B (Code Specialist)": "Qwen/CodeQwen1.5-7B-Chat",
};

// Fallback chain if primary model is rate-limited
const FALLBACK_MODELS = [
  "Qwen/Qwen2.5-3B-Instruct",
  "Qwen/Qwen2.5-0.5B-Instruct",
];

async function callHF(
  hfToken: string,
  modelId: string,
  messages: any[],
  temperature: number,
  max_tokens: number,
  tools?: any[]
): Promise<{ data: any; status: number }> {
  const body: any = {
    model: modelId,
    messages,
    temperature,
    max_tokens: max_tokens,
  };
  // Only include tools if the model supports it
  if (tools && tools.length > 0) {
    body.tools = tools;
    body.tool_choice = "auto";
  }

  const response = await fetch(
    "https://api-inference.huggingface.co/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${hfToken}`,
      },
      body: JSON.stringify(body),
    }
  );

  const data = await response.json();
  return { data, status: response.status };
}

export async function POST(req: NextRequest) {
  try {
    const { messages, model, temperature, max_tokens, tool_results: clientToolResults } = await req.json();

    const hfToken = process.env.HF_TOKEN;
    if (!hfToken) {
      return NextResponse.json(
        { error: "HF_TOKEN environment variable is not set on the server." },
        { status: 500 }
      );
    }

    const requestedModelId = MODEL_MAP[model] || "Qwen/Qwen2.5-7B-Instruct";
    const temp = temperature ?? 0.7;
    const tokens = max_tokens ?? 2048;

    // ── If client sent tool results (quick tools), inject into system message ──
    let finalMessages = messages;
    if (clientToolResults && clientToolResults.length > 0) {
      finalMessages = messages.map((m: any, i: number) => {
        if (i === 0 && m.role === "system") {
          const toolLines = clientToolResults
            .map((t: { name: string; result: string }) => `- ${t.name}: ${t.result}`)
            .join("\n");
          return {
            ...m,
            content:
              m.content +
              `\n\n[Tool Results]\n${toolLines}\nUse these results to answer accurately.`,
          };
        }
        return m;
      });
    }

    // ── Try primary model, fallback on rate limit (429) or model loading (503) ──
    let modelId = requestedModelId;
    let modelsToTry = [modelId, ...FALLBACK_MODELS.filter(m => m !== modelId)];
    let responseData: any = null;
    let lastError = '';
    let usedModel = modelId;

    for (const tryModel of modelsToTry) {
      const { data, status } = await callHF(hfToken, tryModel, finalMessages, temp, tokens);

      if (status === 200 && data.choices?.[0]) {
        responseData = data;
        usedModel = tryModel;
        break;
      }

      // Rate limited or model loading — try next
      if (status === 429 || status === 503) {
        lastError = `Model ${tryModel} is ${status === 429 ? 'rate-limited' : 'loading'} (${status}). Trying fallback...`;
        continue;
      }

      // Other errors — don't fallback, return the error
      if (data.error) {
        return NextResponse.json(
          { error: `HF API error (${status}): ${JSON.stringify(data.error)}` },
          { status }
        );
      }
      lastError = `HF API error (${status}): ${JSON.stringify(data)}`;
    }

    if (!responseData) {
      return NextResponse.json(
        { error: `All models unavailable. ${lastError}` },
        { status: 503 }
      );
    }

    const reply = responseData.choices[0].message?.content || "No response generated.";

    // ── Note: if model returned tool_calls, execute them and loop ──
    const toolCalls = responseData.choices[0].message?.tool_calls;
    if (toolCalls && toolCalls.length > 0) {
      // Append assistant message with tool_calls
      finalMessages.push(responseData.choices[0].message);

      // Execute each tool and append results
      for (const tc of toolCalls) {
        const args = JSON.parse(tc.function.arguments || '{}');
        const result = executeTool({ name: tc.function.name, arguments: args });
        finalMessages.push({
          role: 'tool',
          tool_call_id: tc.id,
          content: result,
        });
      }

      // Second call with tool results
      const { data: data2, status: status2 } = await callHF(hfToken, usedModel, finalMessages, temp, tokens);
      if (status2 === 200 && data2.choices?.[0]) {
        const finalReply = data2.choices[0].message?.content || "No response after tool use.";
        return NextResponse.json({
          content: finalReply,
          model: usedModel,
          tool_used: true,
        });
      }

      // If second call fails, return the first reply anyway
      return NextResponse.json({
        content: reply,
        model: usedModel,
      });
    }

    return NextResponse.json({
      content: reply,
      model: usedModel,
    });
  } catch (error) {
    return NextResponse.json(
      { error: `Server error: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}

import { NextRequest, NextResponse } from "next/server";
import {
  TOOL_DEFINITIONS,
  executeTool,
  detectToolsClient,
} from "@/lib/tools";

const MODEL_MAP: Record<string, string> = {
  "Qwen 2.5 7B (Best Quality)": "Qwen/Qwen2.5-7B-Instruct",
  "Qwen 2.5 3B (Balanced)": "Qwen/Qwen2.5-3B-Instruct",
  "Qwen 2.5 0.5B (Fastest)": "Qwen/Qwen2.5-0.5B-Instruct",
  "CodeQwen 7B (Code Specialist)": "Qwen/CodeQwen1.5-7B-Chat",
};

const FALLBACK_CHAIN = [
  "Qwen/Qwen2.5-3B-Instruct",
  "Qwen/Qwen2.5-0.5B-Instruct",
];

export async function POST(req: NextRequest) {
  try {
    const {
      messages,
      model,
      temperature,
      max_tokens,
      stream: wantStream,
      tool_results: clientToolResults,
    } = await req.json();

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

    // ── Build final messages (inject client-side tool results if any) ──
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

    // ── Streaming mode ──
    if (wantStream) {
      return handleStream(hfToken, requestedModelId, finalMessages, temp, tokens);
    }

    // ── Non-streaming mode with tool loop ──
    return handleNonStream(
      hfToken,
      requestedModelId,
      finalMessages,
      temp,
      tokens
    );
  } catch (error) {
    return NextResponse.json(
      {
        error: `Server error: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 }
    );
  }
}

// ━━━ Non-Streaming with Tool Loop + Fallback ━━━
async function handleNonStream(
  hfToken: string,
  requestedModelId: string,
  messages: any[],
  temperature: number,
  max_tokens: number
) {
  const modelsToTry = [
    requestedModelId,
    ...FALLBACK_CHAIN.filter((m) => m !== requestedModelId),
  ];

  for (const modelId of modelsToTry) {
    const { data, status } = await callHF(
      hfToken,
      modelId,
      messages,
      temperature,
      max_tokens,
      true
    );

    if (status === 429 || status === 503) continue;
    if (status !== 200) {
      return NextResponse.json(
        { error: `HF API error (${status}): ${JSON.stringify(data)}` },
        { status }
      );
    }

    const choice = data.choices?.[0];
    const reply = choice?.message?.content || "No response generated.";

    // Handle tool calls
    const toolCalls = choice?.message?.tool_calls;
    if (toolCalls && toolCalls.length > 0) {
      const toolMessages = [...messages, choice.message];

      for (const tc of toolCalls) {
        const args = JSON.parse(tc.function.arguments || "{}");
        const result = await executeTool({
          name: tc.function.name,
          arguments: args,
        });
        toolMessages.push({
          role: "tool",
          tool_call_id: tc.id,
          content: result,
        });
      }

      // Second call with tool results
      const { data: data2, status: status2 } = await callHF(
        hfToken,
        modelId,
        toolMessages,
        temperature,
        max_tokens,
        false
      );

      if (status2 === 200 && data2.choices?.[0]) {
        return NextResponse.json({
          content: data2.choices[0].message?.content || "No response after tool use.",
          model: modelId,
          tool_used: true,
        });
      }

      return NextResponse.json({ content: reply, model: modelId });
    }

    return NextResponse.json({ content: reply, model: modelId });
  }

  return NextResponse.json(
    { error: "All models are currently unavailable. Please try again in a moment." },
    { status: 503 }
  );
}

// ━━━ Streaming ━━━
async function handleStream(
  hfToken: string,
  requestedModelId: string,
  messages: any[],
  temperature: number,
  max_tokens: number
) {
  const modelsToTry = [
    requestedModelId,
    ...FALLBACK_CHAIN.filter((m) => m !== requestedModelId),
  ];

  for (const modelId of modelsToTry) {
    try {
      const response = await fetch(
        "https://api-inference.huggingface.co/v1/chat/completions",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${hfToken}`,
          },
          body: JSON.stringify({
            model: modelId,
            messages,
            temperature,
            max_tokens,
            stream: true,
          }),
        }
      );

      if (response.status === 429 || response.status === 503) continue;

      if (!response.ok || !response.body) {
        const errText = await response.text();
        // If last model also failed, return error as SSE
        if (modelId === modelsToTry[modelsToTry.length - 1]) {
          const errorStream = new ReadableStream({
            start(controller) {
              controller.enqueue(
                `data: ${JSON.stringify({ error: `HF API error: ${errText}` })}\n\n`
              );
              controller.close();
            },
          });
          return new NextResponse(errorStream, {
            headers: {
              "Content-Type": "text/event-stream",
              "Cache-Control": "no-cache",
              "Connection": "keep-alive",
            },
          });
        }
        continue;
      }

      // Send model name as first event
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async start(controller) {
          try {
            controller.enqueue(
              `data: ${JSON.stringify({ model: modelId })}\n\n`
            );
            const reader = response.body!.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split("\n");
              buffer = lines.pop() || "";

              for (const line of lines) {
                if (line.startsWith("data: ")) {
                  const data = line.slice(6).trim();
                  if (data === "[DONE]") {
                    controller.enqueue(`data: [DONE]\n\n`);
                    continue;
                  }
                  try {
                    JSON.parse(data); // validate
                    controller.enqueue(`data: ${data}\n\n`);
                  } catch {
                    // skip invalid JSON
                  }
                }
              }
            }

            if (buffer.startsWith("data: ")) {
              controller.enqueue(`${buffer}\n\n`);
            }
            controller.enqueue("data: [DONE]\n\n");
            controller.close();
          } catch (err) {
            controller.error(err);
          }
        },
      });

      return new NextResponse(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          "Connection": "keep-alive",
        },
      });
    } catch {
      continue;
    }
  }

  return NextResponse.json(
    { error: "All models are currently unavailable." },
    { status: 503 }
  );
}

// ━━━ HF API Call Helper ━━━
async function callHF(
  hfToken: string,
  modelId: string,
  messages: any[],
  temperature: number,
  max_tokens: number,
 tools: boolean
) {
 const body: any = {
    model: modelId,
    messages,
    temperature,
    max_tokens,
  };
  if (tools) {
    body.tools = TOOL_DEFINITIONS;
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

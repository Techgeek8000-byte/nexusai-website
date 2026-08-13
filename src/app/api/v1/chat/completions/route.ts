import { NextRequest, NextResponse } from "next/server";

const MODEL_MAP: Record<string, string> = {
  "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
  "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
  "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
  "Qwen/CodeQwen1.5-7B-Chat": "Qwen/CodeQwen1.5-7B-Chat",
  1: "Qwen/Qwen2.5-7B-Instruct",
  2: "Qwen/Qwen2.5-3B-Instruct",
  3: "Qwen/Qwen2.5-0.5B-Instruct",
  4: "Qwen/CodeQwen1.5-7B-Chat",
};

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { messages, model, temperature, max_tokens, stream } = body;

    const hfToken = process.env.HF_TOKEN;
    if (!hfToken) {
      return NextResponse.json(
        { error: { message: "API key not configured.", type: "server_error", code: "api_key_missing" } },
        { status: 500 }
      );
    }

    const modelId = MODEL_MAP[model] || "Qwen/Qwen2.5-7B-Instruct";

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
          temperature: temperature ?? 0.7,
          max_tokens: max_tokens ?? 2048,
          stream: stream ?? false,
        }),
      }
    );

    if (!response.ok) {
      const errText = await response.text();
      return NextResponse.json(
        { error: { message: `HF API error (${response.status}): ${errText}`, type: "upstream_error", code: "hf_api_error" } },
        { status: response.status }
      );
    }

    // Stream passthrough
    if (stream) {
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async start(controller) {
          const reader = response.body?.getReader();
          if (!reader) { controller.close(); return; }
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) { controller.close(); break; }
              controller.enqueue(value);
            }
          } catch (err) {
            controller.error(err);
          }
        },
      });
      return new NextResponse(stream, {
        headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" },
      });
    }

    // Non-stream: transform to OpenAI format
    const data = await response.json();
    const choice = data.choices?.[0];
    return NextResponse.json({
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: modelId,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: choice?.message?.content || "",
          },
          finish_reason: choice?.finish_reason || "stop",
        },
      ],
      usage: {
        prompt_tokens: data.usage?.prompt_tokens || 0,
        completion_tokens: data.usage?.completion_tokens || 0,
        total_tokens: data.usage?.total_tokens || 0,
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: { message: `Server error: ${error instanceof Error ? error.message : String(error)}`, type: "server_error" } },
      { status: 500 }
    );
  }
}

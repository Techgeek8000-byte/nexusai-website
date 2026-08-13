import { NextRequest } from "next/server";

const MODEL_MAP: Record<string, string> = {
  "Qwen 2.5 7B (Best Quality)": "Qwen/Qwen2.5-7B-Instruct",
  "Qwen 2.5 3B (Balanced)": "Qwen/Qwen2.5-3B-Instruct",
  "Qwen 2.5 0.5B (Fastest)": "Qwen/Qwen2.5-0.5B-Instruct",
  "CodeQwen 7B (Code Specialist)": "Qwen/CodeQwen1.5-7B-Chat",
};

export async function POST(req: NextRequest) {
  try {
    const { messages, model, temperature, max_tokens, tool_results } = await req.json();

    const hfToken = process.env.HF_TOKEN;
    if (!hfToken) {
      return new Response(
        JSON.stringify({ error: "HF_TOKEN not configured on server." }),
        { status: 500, headers: { "Content-Type": "application/json" } }
      );
    }

    // Build messages with tool results injected
    const finalMessages = messages.map(
      (m: { role: string; content: string }, i: number) => {
        if (
          i === 0 &&
          m.role === "system" &&
          tool_results &&
          tool_results.length > 0
        ) {
          const lines = tool_results
            .map((t: { name: string; result: string }) => `- ${t.name}: ${t.result}`)
            .join("\n");
          return {
            ...m,
            content:
              m.content +
              `\n\n[Tool Results]\n${lines}\n[End Tool Results]`,
          };
        }
        return m;
      }
    );

    const modelId = MODEL_MAP[model] || "Qwen/Qwen2.5-7B-Instruct";

    const hfResponse = await fetch(
      "https://api-inference.huggingface.co/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${hfToken}`,
        },
        body: JSON.stringify({
          model: modelId,
          messages: finalMessages,
          temperature: temperature ?? 0.7,
          max_tokens: max_tokens ?? 2048,
          stream: true,
        }),
      }
    );

    if (!hfResponse.ok) {
      const errText = await hfResponse.text();
      return new Response(
        JSON.stringify({
          error: `HF API error (${hfResponse.status}): ${errText}`,
        }),
        { status: hfResponse.status, headers: { "Content-Type": "application/json" } }
      );
    }

    // Proxy the SSE stream from HF to the client
    const stream = new ReadableStream({
      async start(controller) {
        const reader = hfResponse.body?.getReader();
        if (!reader) {
          controller.close();
          return;
        }
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            controller.enqueue(value);
          }
        } catch (e) {
          controller.error(e);
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({
        error: `Server error: ${error instanceof Error ? error.message : String(error)}`,
      }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}

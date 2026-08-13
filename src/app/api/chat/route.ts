import { NextRequest, NextResponse } from "next/server";

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
      return NextResponse.json(
        { error: "HF_TOKEN environment variable is not set on the server." },
        { status: 500 }
      );
    }

    // Inject tool results into system message if present
    const finalMessages = messages.map((m: { role: string; content: string }, i: number) => {
      if (i === 0 && m.role === "system" && tool_results && tool_results.length > 0) {
        const toolLines = tool_results
          .map((t: { name: string; result: string }) => `- ${t.name}: ${t.result}`)
          .join("\n");
        return {
          ...m,
          content:
            m.content +
            `\n\n[Auto-detected Tool Results]\n${toolLines}\nUse these results to answer accurately.\n[End Tool Results]`,
        };
      }
      return m;
    });

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
          messages: finalMessages,
          temperature: temperature ?? 0.7,
          max_tokens: max_tokens ?? 2048,
        }),
      }
    );

    if (!response.ok) {
      const errText = await response.text();
      return NextResponse.json(
        { error: `HF API error (${response.status}): ${errText}` },
        { status: response.status }
      );
    }

    const data = await response.json();
    const reply = data.choices?.[0]?.message?.content || "No response generated.";

    return NextResponse.json({ content: reply });
  } catch (error) {
    return NextResponse.json(
      { error: `Server error: ${error instanceof Error ? error.message : String(error)}` },
      { status: 500 }
    );
  }
}

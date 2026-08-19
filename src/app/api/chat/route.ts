import { NextRequest, NextResponse } from "next/server";
import {
  TOOL_DEFINITIONS,
  executeTool,
  detectToolsClient,
} from "@/lib/tools";
import { selfAnalyzeResponse, shouldAnalyze } from "@/lib/self-analyze";
import {
  cacheImprovedResponse,
  lookupCachedResponse,
  normalizeQuery,
} from "@/lib/cloud-cache";

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
      self_analyze: wantAnalysis,
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

    // Extract the user's query (last user message)
    const lastUserMsg = [...messages].reverse().find((m: any) => m.role === "user");
    const userQuery = lastUserMsg?.content || "";

    // ── Cloud cache: check if we have an improved version already ──
    if (wantAnalysis !== false) {
      const cached = await lookupCachedResponse(userQuery);
      if (cached.found && cached.response) {
        // Return cached improved response directly
        if (wantStream) {
          return streamText(cached.response, requestedModelId, {
            cached: true,
            qualityScore: cached.qualityScore ?? 80,
          });
        }
        return NextResponse.json({
          content: cached.response,
          model: requestedModelId,
          cache_hit: true,
          quality_score: cached.qualityScore,
          was_improved: cached.wasImproved,
        });
      }
    }

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
      return handleStream(hfToken, requestedModelId, finalMessages, temp, tokens, userQuery, wantAnalysis !== false);
    }

    // ── Non-streaming mode with tool loop + self-analysis + fallback ──
    return handleNonStream(hfToken, requestedModelId, finalMessages, temp, tokens, userQuery, wantAnalysis !== false);
  } catch (error) {
    return NextResponse.json(
      {
        error: `Server error: ${error instanceof Error ? error.message : String(error)}`,
      },
      { status: 500 }
    );
  }
}

// ━━━ Non-Streaming with Tool Loop + Self-Analysis + Fallback ━━━
async function handleNonStream(
  hfToken: string,
  requestedModelId: string,
  messages: any[],
  temperature: number,
  max_tokens: number,
  userQuery: string,
  runAnalysis: boolean
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
    let finalReply = reply;
    let toolUsed = false;

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
        finalReply = data2.choices[0].message?.content || finalReply;
        toolUsed = true;
      }
    }

    // ── Self-Analysis: review and auto-correct ──
    let analysisResult: Awaited<ReturnType<typeof selfAnalyzeResponse>> | null = null;
    if (runAnalysis && shouldAnalyze(finalReply)) {
      try {
        analysisResult = await selfAnalyzeResponse(hfToken, userQuery, finalReply, modelId);

        // If improved, cache the better version in the cloud
        if (analysisResult && analysisResult.improved) {
          // Fire and forget — don't block the response
          cacheImprovedResponse(
            userQuery,
            analysisResult.finalResponse,
            analysisResult.qualityScore,
            modelId,
            analysisResult.issues.map((i) => i.description)
          ).catch(() => {});
        }
      } catch {
        // Self-analysis failure should never break the response
        analysisResult = null;
      }
    }

    return NextResponse.json({
      content: analysisResult?.finalResponse || finalReply,
      model: modelId,
      tool_used: toolUsed,
      self_analysis: analysisResult
        ? {
            improved: analysisResult.improved,
            quality_score: analysisResult.qualityScore,
            issues: analysisResult.issues,
            analysis_time_ms: analysisResult.analysisTimeMs,
          }
        : undefined,
    });
  }

  return NextResponse.json(
    { error: "All models are currently unavailable. Please try again in a moment." },
    { status: 503 }
  );
}

// ━━━ Streaming with Post-Stream Analysis Flag ━━━
async function handleStream(
  hfToken: string,
  requestedModelId: string,
  messages: any[],
  temperature: number,
  max_tokens: number,
  userQuery: string,
  runAnalysis: boolean
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

      // ── Stream with post-completion self-analysis ──
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
            let fullContent = "";

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
                    const parsed = JSON.parse(data);
                    const delta = parsed.choices?.[0]?.delta?.content;
                    if (delta) {
                      fullContent += delta;
                    }
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

            // ── Post-stream: trigger self-analysis in background ──
            // Send analysis event after stream completes
            if (runAnalysis && shouldAnalyze(fullContent)) {
              try {
                const analysisResult = await selfAnalyzeResponse(hfToken, userQuery, fullContent, modelId);

                if (analysisResult.improved) {
                  // Cache the improved version for future queries
                  cacheImprovedResponse(
                    userQuery,
                    analysisResult.finalResponse,
                    analysisResult.qualityScore,
                    modelId,
                    analysisResult.issues.map((i) => i.description)
                  ).catch(() => {});

                  // Send the improved version as a special event
                  controller.enqueue(
                    `data: ${JSON.stringify({
                      type: 'self_improved',
                      improved_response: analysisResult.finalResponse,
                      quality_score: analysisResult.qualityScore,
                      issues: analysisResult.issues,
                      analysis_time_ms: analysisResult.analysisTimeMs,
                    })}\n\n`
                  );
                } else if (analysisResult.issues.length > 0) {
                  // Response passed review (good enough) but had minor notes
                  controller.enqueue(
                    `data: ${JSON.stringify({
                      type: 'self_analysis',
                      quality_score: analysisResult.qualityScore,
                      issues: analysisResult.issues,
                      analysis_time_ms: analysisResult.analysisTimeMs,
                    })}\n\n`
                  );
                }
              } catch {
                // Analysis failure — stream already complete, ignore
              }
            }

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

// ━━━ Stream a cached response as SSE ━━━
function streamText(
  text: string,
  modelId: string,
  meta: { cached: boolean; qualityScore: number }
) {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      // Send model
      controller.enqueue(
        `data: ${JSON.stringify({ model: modelId })}\n\n`
      );
      // Send cache hit notification
      controller.enqueue(
        `data: ${JSON.stringify({
          type: 'cache_hit',
          quality_score: meta.qualityScore,
        })}\n\n`
      );
      // Stream text in small chunks for visual effect
      const chunkSize = 8;
      for (let i = 0; i < text.length; i += chunkSize) {
        const chunk = text.slice(i, i + chunkSize);
        controller.enqueue(
          `data: ${JSON.stringify({
            choices: [{ delta: { content: chunk } }],
          })}\n\n`
        );
      }
      controller.enqueue("data: [DONE]\n\n");
      controller.close();
    },
  });

  return new NextResponse(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
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

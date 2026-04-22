export interface OpencodeSnippetOptions {
  model: string;
  localPort: number;
  maxModelLen?: number;
  toolCall?: boolean;
  reasoning?: boolean;
}

export function formatOpencodeSnippet(opts: OpencodeSnippetOptions): string {
  const { model, localPort, maxModelLen, toolCall, reasoning } = opts;
  const context = maxModelLen ?? 4096;

  const modelEntry: Record<string, unknown> = {
    name: `${model} (Isambard)`,
    limit: { context, output: context },
  };
  if (toolCall) modelEntry["tool_call"] = true;
  if (reasoning) modelEntry["reasoning"] = true;

  const snippet = {
    "$schema": "https://opencode.ai/config.json",
    provider: {
      "isambard-vllm": {
        npm: "@ai-sdk/openai-compatible",
        name: "Isambard vLLM Server",
        options: {
          baseURL: `http://localhost:${localPort}/v1`,
          apiKey: "EMPTY",
        },
        models: {
          [model]: modelEntry,
        },
      },
    },
  };

  return JSON.stringify(snippet, null, 2);
}

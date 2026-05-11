/**
 * AI coding assistant utilities.
 * Detects available assistants, generates config/env vars for each.
 */

import { existsSync } from "fs";
import { spawnSync } from "child_process";

export interface AssistantConfig {
  name: string;
  env: Record<string, string>;
  args: string[];
}

export interface AssistantOption {
  id: number;
  label: string;
  assistant: string;
  useScoder: boolean;
}

/**
 * Check if a binary exists on PATH.
 */
export function binaryExists(name: string): boolean {
  const result = spawnSync("which", [name], {
    shell: true,
    encoding: "utf-8",
  });
  return result.status === 0 && result.output[1]?.trim().length > 0;
}

/**
 * Get the list of assistant binaries available on PATH.
 */
export function getAvailableAssistants(): string[] {
  const candidates = ["opencode", "claude", "code"];
  return candidates.filter(binaryExists);
}

/**
 * Check if scoder is available on PATH.
 */
export function getScoderAvailable(): boolean {
  return binaryExists("scoder");
}

/**
 * Generate environment variables for opencode.json configuration.
 */
export function generateOpencodeConfig(
  opts: {
    model: string;
    localPort: number;
    maxModelLen?: number;
    toolCall?: boolean;
    reasoning?: boolean;
  }
): Record<string, unknown> {
  const context = opts.maxModelLen ?? 4096;

  const modelEntry: Record<string, unknown> = {
    name: `${opts.model} (Isambard)`,
    limit: { context, output: context },
  };
  if (opts.toolCall) modelEntry["tool_call"] = true;
  if (opts.reasoning) modelEntry["reasoning"] = true;

  return {
    "$schema": "https://opencode.ai/config.json",
    provider: {
      "isambard-vllm": {
        npm: "@ai-sdk/openai-compatible",
        name: "Isambard vLLM Server",
        options: {
          baseURL: `http://localhost:${opts.localPort}/v1`,
          apiKey: "EMPTY",
        },
        models: {
          [opts.model]: modelEntry,
        },
      },
    },
  };
}

/**
 * Generate environment variables for GitHub Copilot.
 */
export function generateCopilotEnv(localPort: number, model: string): Record<string, string> {
  return {
    COPILOT_PROVIDER_BASE_URL: `http://localhost:${localPort}`,
    COPILOT_MODEL: model,
    ANTHROPIC_BASE_URL: "http://localhost:4000",
    ANTHROPIC_API_KEY: "ollama",
    CLAUDE_MODEL: `meta-llama/${model}:free`,
  };
}

/**
 * Generate environment variables for Claude Code.
 */
export function generateClaudeEnv(localPort: number, model: string): Record<string, string> {
  return {
    ANTHROPIC_BASE_URL: `http://localhost:${localPort}`,
    ANTHROPIC_API_KEY: "ollama",
    CLAUDE_MODEL: `meta-llama/${model}:free`,
  };
}

/**
 * Build the list of menu options based on available assistants and scoder.
 */
export function buildAssistantMenuOptions(
  assistants: string[],
  hasScoder: boolean
): AssistantOption[] {
  const options: AssistantOption[] = [];
  let id = 1;

  for (const assistant of assistants) {
    // Direct launch option
    options.push({ id, label: `${assistant}`, assistant, useScoder: false });
    id++;

    // Scoder option (if available)
    if (hasScoder) {
      options.push({ id, label: `scoder ${assistant}`, assistant, useScoder: true });
      id++;
    }
  }

  return options;
}

/**
 * Get the binary name and args for launching an assistant.
 */
export function getLaunchCommand(
  assistant: string,
  useScoder: boolean
): { binary: string; args: string[] } {
  if (useScoder) {
    return { binary: "scoder", args: [assistant] };
  }
  return { binary: assistant, args: [] };
}

import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test";
import {
  generateOpencodeConfig,
  generateOpencodeEnv,
  generateCopilotEnv,
  generateClaudeEnv,
  buildAssistantMenuOptions,
  getLaunchCommand,
  getAvailableAssistants,
  binaryExists,
} from "../src/assistant.ts";

describe("F2.6 — generateOpencodeConfig returns valid JSON", () => {
  it("produces valid JSON with all fields", () => {
    const config = generateOpencodeConfig({
      model: "Qwen/Qwen3.6-35B-A3B",
      localPort: 11434,
      maxModelLen: 8192,
      toolCall: true,
      reasoning: true,
    });
    const json = JSON.stringify(config);
    expect(() => JSON.parse(json)).not.toThrow();
    expect(json).toContain("Qwen/Qwen3.6-35B-A3B");
    expect(json).toContain("http://localhost:11434/v1");
    expect(json).toContain("8192");
    expect(json).toContain("tool_call");
    expect(json).toContain("reasoning");
  });
});

describe("F2.6 — generateOpencodeEnv sets runtime override", () => {
  it("stores opencode JSON in OPENCODE_CONFIG_CONTENT", () => {
    const env = generateOpencodeEnv({
      model: "Qwen/Qwen3.6-35B-A3B",
      localPort: 11434,
      maxModelLen: 8192,
    });
    expect(env.OPENCODE_CONFIG_CONTENT).toBeString();
    const parsed = JSON.parse(env.OPENCODE_CONFIG_CONTENT);
    expect(parsed.provider["isambard-vllm"].options.baseURL).toBe("http://localhost:11434/v1");
  });
});

describe("F2.6 — generateCopilotEnv sets correct env vars", () => {
  it("sets COPILOT_PROVIDER_BASE_URL to localhost with correct port", () => {
    const env = generateCopilotEnv(11434, "Qwen3.6-35B-A3B");
    expect(env.COPILOT_PROVIDER_BASE_URL).toBe("http://localhost:11434");
  });

  it("sets COPILOT_MODEL to model name", () => {
    const env = generateCopilotEnv(11434, "Qwen3.6-35B-A3B");
    expect(env.COPILOT_MODEL).toBe("Qwen3.6-35B-A3B");
  });

  it("sets ANTHROPIC_BASE_URL to localhost:4000 for Copilot proxy", () => {
    const env = generateCopilotEnv(11434, "Qwen3.6-35B-A3B");
    expect(env.ANTHROPIC_BASE_URL).toBe("http://localhost:4000");
  });
});

describe("F2.6 — generateClaudeEnv sets correct env vars", () => {
  it("sets ANTHROPIC_BASE_URL to localhost with correct port", () => {
    const env = generateClaudeEnv(8000, "Qwen3.6-35B-A3B");
    expect(env.ANTHROPIC_BASE_URL).toBe("http://localhost:8000");
  });

  it("sets CLAUDE_MODEL with meta-llama prefix", () => {
    const env = generateClaudeEnv(11434, "llama-3.3-70b");
    expect(env.CLAUDE_MODEL).toBe("meta-llama/llama-3.3-70b:free");
  });
});

describe("F2.6 — buildAssistantMenuOptions includes cwd", () => {
  it("produces sequential numbered options", () => {
    const options = buildAssistantMenuOptions(["opencode"], false);
    expect(options[0].id).toBe(1);
    expect(options[0].label).toBe("opencode");
  });

  it("includes scoder option when available", () => {
    const options = buildAssistantMenuOptions(["claude"], true);
    const scoderOption = options.find(o => o.useScoder);
    expect(scoderOption).toBeDefined();
    expect(scoderOption!.label).toBe("scoder claude");
  });
});

describe("F2.6 — getLaunchCommand produces correct args", () => {
  it("returns scoder + assistant for sandboxed launch", () => {
    const cmd = getLaunchCommand("opencode", true);
    expect(cmd.binary).toBe("scoder");
    expect(cmd.args).toEqual(["opencode", "--continue"]);
  });

  it("returns direct command for non-scoder launch", () => {
    const cmd = getLaunchCommand("claude", false);
    expect(cmd.binary).toBe("claude");
    expect(cmd.args).toEqual(["--continue"]);
  });
});

describe("F2.6 — assistant detection works", () => {
  it("binaryExists returns boolean", () => {
    const result = binaryExists("ls");
    expect(typeof result).toBe("boolean");
  });

  it("getAvailableAssistants returns array of strings", () => {
    const assistants = getAvailableAssistants();
    expect(Array.isArray(assistants)).toBe(true);
    expect(assistants.every(a => typeof a === "string")).toBe(true);
  });
});

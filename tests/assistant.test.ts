import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import {
  binaryExists,
  getAvailableAssistants,
  getScoderAvailable,
  generateOpencodeConfig,
  generateCopilotEnv,
  generateClaudeEnv,
  buildAssistantMenuOptions,
  getLaunchCommand,
} from "../src/assistant.ts";

describe("binaryExists", () => {
  it("returns true for 'ls' which is always on PATH", () => {
    expect(binaryExists("ls")).toBe(true);
  });

  it("returns false for a non-existent command", () => {
    expect(binaryExists("nonexistent-command-xyz-12345")).toBe(false);
  });
});

describe("generateOpencodeConfig", () => {
  it("includes the correct schema", () => {
    const config = generateOpencodeConfig({ model: "Qwen/Qwen2.5-0.5B-Instruct", localPort: 11434 });
    expect(JSON.stringify(config)).toContain("$schema");
    expect(JSON.stringify(config)).toContain("https://opencode.ai/config.json");
  });

  it("includes baseURL with correct port", () => {
    const config = generateOpencodeConfig({ model: "test/model", localPort: 8080 });
    expect(JSON.stringify(config)).toContain("http://localhost:8080/v1");
  });

  it("includes model name in entry", () => {
    const config = generateOpencodeConfig({ model: "Qwen/Qwen3.6-35B-A3B", localPort: 11434 });
    expect(JSON.stringify(config)).toContain("Qwen/Qwen3.6-35B-A3B");
  });

  it("includes maxModelLen as context limit", () => {
    const config = generateOpencodeConfig({ model: "test", localPort: 11434, maxModelLen: 2048 });
    expect(JSON.stringify(config)).toContain("2048");
  });

  it("includes tool_call when toolCall is true", () => {
    const config = generateOpencodeConfig({ model: "test", localPort: 11434, toolCall: true });
    expect(JSON.stringify(config)).toContain('"tool_call":true');
  });

  it("excludes tool_call when toolCall is false", () => {
    const config = generateOpencodeConfig({ model: "test", localPort: 11434, toolCall: false });
    expect(JSON.stringify(config)).not.toContain('"tool_call"');
  });

  it("includes reasoning when reasoning is true", () => {
    const config = generateOpencodeConfig({ model: "test", localPort: 11434, reasoning: true });
    expect(JSON.stringify(config)).toContain('"reasoning":true');
  });

  it("defaults maxModelLen to 4096", () => {
    const config = generateOpencodeConfig({ model: "test", localPort: 11434 });
    expect(JSON.stringify(config)).toContain("4096");
  });
});

describe("generateCopilotEnv", () => {
  it("includes COPILOT_PROVIDER_BASE_URL", () => {
    const env = generateCopilotEnv(11434, "test-model");
    expect(env.COPILOT_PROVIDER_BASE_URL).toBe("http://localhost:11434");
  });

  it("includes COPILOT_MODEL", () => {
    const env = generateCopilotEnv(11434, "test-model");
    expect(env.COPILOT_MODEL).toBe("test-model");
  });

  it("includes ANTHROPIC_BASE_URL", () => {
    const env = generateCopilotEnv(11434, "test-model");
    expect(env.ANTHROPIC_BASE_URL).toBe("http://localhost:4000");
  });

  it("includes ANTHROPIC_API_KEY set to ollama", () => {
    const env = generateCopilotEnv(11434, "test-model");
    expect(env.ANTHROPIC_API_KEY).toBe("ollama");
  });

  it("includes CLAUDE_MODEL with meta-llama prefix", () => {
    const env = generateCopilotEnv(11434, "llama-3.3-70b-instruct");
    expect(env.CLAUDE_MODEL).toBe("meta-llama/llama-3.3-70b-instruct:free");
  });
});

describe("generateClaudeEnv", () => {
  it("includes ANTHROPIC_BASE_URL with correct port", () => {
    const env = generateClaudeEnv(8080, "test-model");
    expect(env.ANTHROPIC_BASE_URL).toBe("http://localhost:8080");
  });

  it("includes ANTHROPIC_API_KEY set to ollama", () => {
    const env = generateClaudeEnv(11434, "test-model");
    expect(env.ANTHROPIC_API_KEY).toBe("ollama");
  });

  it("includes CLAUDE_MODEL with meta-llama prefix", () => {
    const env = generateClaudeEnv(11434, "llama-3.5-70b");
    expect(env.CLAUDE_MODEL).toBe("meta-llama/llama-3.5-70b:free");
  });
});

describe("buildAssistantMenuOptions", () => {
  it("includes both direct and scoder options when scoder available", () => {
    const options = buildAssistantMenuOptions(["opencode", "claude"], true);
    expect(options.length).toBe(4); // 2 assistants × 2 launch modes
    expect(options[0].label).toBe("opencode");
    expect(options[0].useScoder).toBe(false);
    expect(options[1].label).toBe("scoder opencode");
    expect(options[1].useScoder).toBe(true);
    expect(options[2].label).toBe("claude");
    expect(options[3].label).toBe("scoder claude");
  });

  it("includes only direct options when scoder not available", () => {
    const options = buildAssistantMenuOptions(["opencode", "claude"], false);
    expect(options.length).toBe(2);
    expect(options[0].label).toBe("opencode");
    expect(options[0].useScoder).toBe(false);
    expect(options[1].label).toBe("claude");
    expect(options[1].useScoder).toBe(false);
  });

  it("numbers menu options sequentially", () => {
    const options = buildAssistantMenuOptions(["opencode", "claude", "code"], true);
    expect(options.length).toBe(6);
    expect(options[0].id).toBe(1);
    expect(options[1].id).toBe(2);
    expect(options[2].id).toBe(3);
    expect(options[5].id).toBe(6);
  });

  it("handles empty assistant list", () => {
    const options = buildAssistantMenuOptions([], true);
    expect(options.length).toBe(0);
  });
});

describe("getLaunchCommand", () => {
  it("returns scoder command for scoder launch", () => {
    const cmd = getLaunchCommand("claude", true);
    expect(cmd.binary).toBe("scoder");
    expect(cmd.args).toEqual(["claude"]);
  });

  it("returns direct command for non-scoder launch", () => {
    const cmd = getLaunchCommand("opencode", false);
    expect(cmd.binary).toBe("opencode");
    expect(cmd.args).toEqual([]);
  });
});

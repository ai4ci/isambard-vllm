import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import {
  binaryExists,
  getAvailableAssistants,
  getScoderAvailable,
  getSbxAvailable,
  generateOpencodeConfig,
  generateOpencodeEnv,
  generateAssistantEnv,
  generateCopilotEnv,
  generateClaudeEnv,
  getAvailableWrappers,
  buildSandboxName,
  buildSandboxCreateCommand,
  parseSbxSandboxes,
  findMatchingSandbox,
  buildLaunchCommand,
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

describe("generateOpencodeEnv", () => {
  it("sets OPENCODE_CONFIG_CONTENT", () => {
    const env = generateOpencodeEnv({ model: "test/model", localPort: 8080 });
    expect(env.OPENCODE_CONFIG_CONTENT).toBeDefined();
  });

  it("encodes the same config shape used for manual snippets", () => {
    const env = generateOpencodeEnv({ model: "test/model", localPort: 8080, toolCall: true });
    const parsed = JSON.parse(env.OPENCODE_CONFIG_CONTENT);
    expect(parsed.provider["isambard-vllm"].options.baseURL).toBe("http://localhost:8080/v1");
    expect(parsed.provider["isambard-vllm"].models["test/model"].tool_call).toBe(true);
  });
});

describe("generateCopilotEnv", () => {
  it("includes COPILOT_PROVIDER_BASE_URL", () => {
    const env = generateCopilotEnv(11434, "test-model");
    expect(env.COPILOT_PROVIDER_BASE_URL).toBe("http://localhost:11434/v1");
  });

  it("includes COPILOT_MODEL", () => {
    const env = generateCopilotEnv(11434, "test-model");
    expect(env.COPILOT_MODEL).toBe("test-model");
  });

  it("does not include ANTHROPIC_BASE_URL", () => {
    const env = generateCopilotEnv(11434, "test-model");
    expect(env.ANTHROPIC_BASE_URL).toBeUndefined();
  });

  it("does not include ANTHROPIC_API_KEY", () => {
    const env = generateCopilotEnv(11434, "test-model");
    expect(env.ANTHROPIC_API_KEY).toBeUndefined();
  });

  it("does not include CLAUDE_MODEL", () => {
    const env = generateCopilotEnv(11434, "llama-3.3-70b-instruct");
    expect(env.CLAUDE_MODEL).toBeUndefined();
  });
});

describe("getSbxAvailable", () => {
  it("returns a boolean", () => {
    expect(typeof getSbxAvailable()).toBe("boolean");
  });
});

describe("generateAssistantEnv", () => {
  it("uses OPENCODE_CONFIG_CONTENT for opencode", () => {
    const env = generateAssistantEnv("opencode", { model: "test/model", localPort: 11434 });
    expect(env.OPENCODE_CONFIG_CONTENT).toBeDefined();
  });

  it("switches opencode baseURL host for sandbox launches", () => {
    const env = generateAssistantEnv("opencode", {
      model: "test/model",
      localPort: 11434,
      endpointHost: "host.docker.internal",
    });
    const parsed = JSON.parse(env.OPENCODE_CONFIG_CONTENT);
    expect(parsed.provider["isambard-vllm"].options.baseURL).toBe("http://host.docker.internal:11434/v1");
  });

  it("uses host.docker.internal for claude when requested", () => {
    const env = generateAssistantEnv("claude", {
      model: "test-model",
      localPort: 11434,
      endpointHost: "host.docker.internal",
    });
    expect(env.ANTHROPIC_BASE_URL).toBe("http://host.docker.internal:11434");
    expect(env.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe("test-model");
  });
});

describe("buildSandboxName", () => {
  it("uses assistant name plus workspace basename", () => {
    expect(buildSandboxName("copilot", "/home/test/isambard-vllm")).toBe("copilot-isambard-vllm");
  });
});

describe("buildSandboxCreateCommand", () => {
  it("renders sbx create with derived sandbox name", () => {
    expect(buildSandboxCreateCommand("copilot", "/home/test/isambard-vllm"))
      .toContain("sbx create --name 'copilot-isambard-vllm' copilot '/home/test/isambard-vllm'");
  });
});

describe("getAvailableWrappers", () => {
  it("includes direct and scoder for local assistant binaries", () => {
    expect(getAvailableWrappers("claude", ["claude"], true, false)).toEqual(["none", "scoder"]);
  });

  it("includes sbx even when the assistant is not installed locally", () => {
    expect(getAvailableWrappers("copilot", [], false, true)).toEqual(["sbx"]);
  });
});

describe("parseSbxSandboxes", () => {
  it("parses sbx ls json output", () => {
    const sandboxes = parseSbxSandboxes(JSON.stringify([
      {
        name: "copilot-isambard-vllm",
        agent: "copilot",
        workspace: "/home/test/isambard-vllm",
        status: "running",
      },
    ]));
    expect(sandboxes[0]?.name).toBe("copilot-isambard-vllm");
    expect(sandboxes[0]?.agent).toBe("copilot");
  });
});

describe("findMatchingSandbox", () => {
  it("finds a sandbox by agent and workspace", () => {
    const sandbox = findMatchingSandbox([
      { name: "copilot-isambard-vllm", agent: "copilot", workspace: "/home/test/isambard-vllm" },
      { name: "claude-other", agent: "claude", workspace: "/home/test/other" },
    ], "copilot", "/home/test/isambard-vllm");
    expect(sandbox?.name).toBe("copilot-isambard-vllm");
  });
});

describe("buildLaunchCommand", () => {
  const opts = {
    assistant: "claude" as const,
    wrapper: "none" as const,
    cwd: "/tmp/my project",
    model: "test-model",
    localPort: 11434,
  };

  it("renders a direct launch command with cwd and env vars", () => {
    const command = buildLaunchCommand(opts);
    expect(command).toContain("cd '/tmp/my project' &&");
    expect(command).toContain("ANTHROPIC_BASE_URL='http://localhost:11434'");
    expect(command).toContain("ANTHROPIC_DEFAULT_SONNET_MODEL='test-model'");
    expect(command).toContain("ANTHROPIC_DEFAULT_OPUS_MODEL='test-model'");
    expect(command).toContain("ANTHROPIC_DEFAULT_HAIKU_MODEL='test-model'");
    expect(command).toContain("CLAUDE_CODE_SUBAGENT_MODEL='test-model'");
    expect(command).toContain(" claude");
    expect(command).not.toContain("claude --continue");
  });

  it("renders a scoder launch command with explicit llm port", () => {
    const command = buildLaunchCommand({ ...opts, assistant: "opencode", wrapper: "scoder" });
    expect(command).toContain("OPENCODE_CONFIG_CONTENT=");
    expect(command).toContain("scoder --llm-port 11434 opencode");
    expect(command).not.toContain("opencode --continue");
  });

  it("renders an sbx launch command with sandbox env injection", () => {
    const command = buildLaunchCommand({
      ...opts,
      assistant: "copilot",
      wrapper: "sbx",
      cwd: "/home/test/isambard-vllm",
    });
    expect(command).toContain("sbx exec -it");
    expect(command).toContain("-w '/home/test/isambard-vllm'");
    expect(command).toContain("-e 'COPILOT_PROVIDER_BASE_URL=http://host.docker.internal:11434/v1'");
    expect(command).toContain("copilot-isambard-vllm copilot --continue");
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

  it("maps all default Anthropic model env vars to the configured model", () => {
    const env = generateClaudeEnv(11434, "llama-3.5-70b");
    expect(env.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe("llama-3.5-70b");
    expect(env.ANTHROPIC_DEFAULT_OPUS_MODEL).toBe("llama-3.5-70b");
    expect(env.ANTHROPIC_DEFAULT_HAIKU_MODEL).toBe("llama-3.5-70b");
    expect(env.CLAUDE_CODE_SUBAGENT_MODEL).toBe("llama-3.5-70b");
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

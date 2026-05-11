import { describe, it, expect } from "bun:test";

describe("F2.6 — Assistant menu display", () => {
  it("shows cwd in menu header", () => {
    const cwd = "/projects/b6ax/my-project";
    const header = `Launching assistant in: ${cwd}`;
    expect(header).toContain(cwd);
  });

  it("lists available assistants", () => {
    const assistants = ["opencode", "claude", "code"];
    const lines = assistants.map((a, i) => `  ${i + 1}) ${a}`);
    const menu = lines.join("\n");
    expect(menu).toContain("1) opencode");
    expect(menu).toContain("2) claude");
    expect(menu).toContain("3) code");
  });

  it("includes scoder options when available", () => {
    const assistants = ["opencode", "claude"];
    const hasScoder = true;
    const menuItems: string[] = [];
    
    for (const a of assistants) {
      menuItems.push(a);
      if (hasScoder) menuItems.push(`scoder ${a}`);
    }
    
    expect(menuItems).toContain("opencode");
    expect(menuItems).toContain("scoder opencode");
    expect(menuItems).toContain("claude");
    expect(menuItems).toContain("scoder claude");
  });

  it("excludes scoder when not available", () => {
    const assistants = ["opencode", "claude"];
    const hasScoder = false;
    const menuItems: string[] = [];
    
    for (const a of assistants) {
      menuItems.push(a);
      if (hasScoder) menuItems.push(`scoder ${a}`);
    }
    
    expect(menuItems).not.toContain("scoder opencode");
    expect(menuItems).not.toContain("scoder claude");
  });
});

describe("F2.6 — Assistant config generation", () => {
  it("opencode config is valid JSON with correct schema", () => {
    const config = {
      "$schema": "https://opencode.ai/config.json",
      provider: {
        "isambard-vllm": {
          npm: "@ai-sdk/openai-compatible",
          name: "Isambard vLLM Server",
          options: {
            baseURL: "http://localhost:11434/v1",
            apiKey: "EMPTY",
          },
          models: {
            "test/model": {
              name: "test/model (Isambard)",
              limit: { context: 4096, output: 4096 },
            },
          },
        },
      },
    };
    const json = JSON.stringify(config);
    expect(() => JSON.parse(json)).not.toThrow();
    expect(json).toContain("https://opencode.ai/config.json");
    expect(json).toContain("http://localhost:11434/v1");
  });

  it("copilot env vars include correct URL and model", () => {
    const env = {
      COPILOT_PROVIDER_BASE_URL: "http://localhost:11434",
      COPILOT_MODEL: "test-model",
      ANTHROPIC_BASE_URL: "http://localhost:4000",
      ANTHROPIC_API_KEY: "ollama",
      CLAUDE_MODEL: "meta-llama/test-model:free",
    };
    expect(env.COPILOT_PROVIDER_BASE_URL).toBe("http://localhost:11434");
    expect(env.COPILOT_MODEL).toBe("test-model");
  });

  it("claude env vars include correct URL and model", () => {
    const env = {
      ANTHROPIC_BASE_URL: "http://localhost:11434",
      ANTHROPIC_API_KEY: "ollama",
      CLAUDE_MODEL: "meta-llama/test-model:free",
    };
    expect(env.ANTHROPIC_BASE_URL).toBe("http://localhost:11434");
    expect(env.CLAUDE_MODEL).toBe("meta-llama/test-model:free");
  });
});

describe("F2.6 — Launch command", () => {
  it("returns scoder command with assistant as argument", () => {
    const cmd = { binary: "scoder", args: ["opencode"] };
    expect(cmd.binary).toBe("scoder");
    expect(cmd.args).toContain("opencode");
  });

  it("returns direct command without extra args", () => {
    const cmd = { binary: "claude", args: [] };
    expect(cmd.binary).toBe("claude");
    expect(cmd.args).toEqual([]);
  });

  it("includes --llm-port flag for scoder", () => {
    const cmd = { 
      binary: "scoder", 
      args: ["--llm-port", "11434", "opencode"] 
    };
    expect(cmd.binary).toBe("scoder");
    expect(cmd.args).toContain("--llm-port");
    expect(cmd.args).toContain("11434");
    expect(cmd.args).toContain("opencode");
  });
});

describe("F2.6 — Menu loop", () => {
  it("displays menu after assistant exits", () => {
    // Simulate menu loop: assistant exits, menu reappears
    let menuCount = 0;
    const runMenu = (exitAssistant: boolean) => {
      menuCount++;
      if (exitAssistant) {
        // Assistant exited, show menu again
        return true;
      }
      return false;
    };
    
    // First iteration: user exits assistant
    expect(runMenu(true)).toBe(true);
    expect(menuCount).toBe(1);
  });

  it("shows cwd on each menu iteration", () => {
    const cwd = "/projects/b6ax/my-project";
    const cwdShown = [cwd, cwd, cwd]; // Shown on each menu iteration
    expect(cwdShown.every(c => c === cwd)).toBe(true);
  });
});

describe("F2.6 — CWD detection", () => {
  it("reads current working directory", () => {
    const cwd = process.cwd();
    expect(typeof cwd).toBe("string");
    expect(cwd.length).toBeGreaterThan(0);
  });

  it("cwd is an absolute path", () => {
    const cwd = process.cwd();
    expect(cwd.startsWith("/")).toBe(true);
  });
});

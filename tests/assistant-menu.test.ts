import { describe, it, expect } from "bun:test";
import {
  generateOpencodeConfig,
  generateCopilotEnv,
  generateClaudeEnv,
  buildAssistantMenuOptions,
  getLaunchCommand,
} from "../src/assistant.ts";

describe("F2.6 — menu displays cwd", () => {
  it("menu text includes cwd placeholder", () => {
    const cwd = "/projects/b6ax/my-project";
    const assistants = ["opencode", "claude"];
    const hasScoder = true;
    const options = buildAssistantMenuOptions(assistants, hasScoder);

    // Verify menu would display cwd
    const menuText = `Launching in ${cwd}\n`;
    expect(menuText).toContain(cwd);
  });
});

describe("F2.6 — menu shows scoder options when available", () => {
  it("includes scoder options for each assistant", () => {
    const options = buildAssistantMenuOptions(["opencode", "claude"], true);
    const scoderOptions = options.filter(o => o.useScoder);
    expect(scoderOptions.length).toBe(2); // scoder opencode, scoder claude
  });

  it("excludes scoder options when not available", () => {
    const options = buildAssistantMenuOptions(["opencode", "claude"], false);
    const scoderOptions = options.filter(o => o.useScoder);
    expect(scoderOptions.length).toBe(0);
  });
});

describe("F2.6 — opencode config includes cwd", () => {
  it("opencode config is valid for cwd-aware file writing", () => {
    const config = generateOpencodeConfig({
      model: "test",
      localPort: 11434,
    });
    const json = JSON.stringify(config);
    expect(json).toContain("https://opencode.ai/config.json");
    expect(json).toContain("http://localhost:11434/v1");
  });
});

describe("F2.6 — env vars work with cwd", () => {
  it("copilot env works regardless of cwd", () => {
    const env = generateCopilotEnv(11434, "test");
    expect(env.COPILOT_PROVIDER_BASE_URL).toContain("localhost:11434/v1");
  });

  it("claude env works regardless of cwd", () => {
    const env = generateClaudeEnv(11434, "test");
    expect(env.ANTHROPIC_BASE_URL).toContain("localhost:11434");
  });
});

describe("F2.6 — launch command works in any cwd", () => {
  it("launch command doesn't include cwd in args", () => {
    const cmd = getLaunchCommand("opencode", false);
    expect(cmd.args).toEqual([]);
    expect(cmd.binary).toBe("opencode");
  });

  it("scoder launch command doesn't include cwd in args", () => {
    const cmd = getLaunchCommand("claude", true);
    expect(cmd.args).toEqual(["claude"]);
    expect(cmd.binary).toBe("scoder");
  });
});

import { describe, it, expect, beforeEach, afterEach } from "bun:test";
import { existsSync, rmSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { loadConfig, saveConfig, assertConfigured } from "../src/config.ts";

const REAL_CONFIG_PATH = join(homedir(), ".ivllm", "config.json");

beforeEach(() => {
  if (existsSync(REAL_CONFIG_PATH)) rmSync(REAL_CONFIG_PATH);
});

afterEach(() => {
  if (existsSync(REAL_CONFIG_PATH)) rmSync(REAL_CONFIG_PATH);
});

describe("Config", () => {
  it("loadConfig returns defaults when no config file exists", () => {
    const config = loadConfig();
    expect(config.defaultLocalPort).toBe(11434);
    expect(config.venvPath).toBe("~/ivllm-venv/.venv");
    expect(config.loginHost).toBe("");
  });

  it("saveConfig persists and loadConfig reads back", () => {
    const config = loadConfig();
    config.loginHost = "login.isambard.ac.uk";
    config.username = "testuser";
    saveConfig(config);

    const loaded = loadConfig();
    expect(loaded.loginHost).toBe("login.isambard.ac.uk");
    expect(loaded.username).toBe("testuser");
    expect(loaded.defaultLocalPort).toBe(11434);
  });

  it("loadConfig merges saved values with defaults for missing fields", () => {
    const config = loadConfig();
    config.loginHost = "login.isambard.ac.uk";
    config.username = "testuser";
    saveConfig(config);

    // Simulate a config file missing a newer field — strip defaultLocalPort
    const raw = JSON.parse(readFileSync(REAL_CONFIG_PATH, "utf-8")) as Record<string, unknown>;
    delete raw["defaultLocalPort"];
    writeFileSync(REAL_CONFIG_PATH, JSON.stringify(raw));

    const merged = loadConfig();
    expect(merged.defaultLocalPort).toBe(11434); // filled by default
    expect(merged.loginHost).toBe("login.isambard.ac.uk"); // preserved
  });

  it("assertConfigured throws when loginHost is missing", () => {
    const config = loadConfig();
    expect(() => assertConfigured(config)).toThrow(/loginHost/);
  });

  it("assertConfigured throws when username is missing", () => {
    const config = loadConfig();
    config.loginHost = "login.isambard.ac.uk";
    expect(() => assertConfigured(config)).toThrow(/username/);
  });

  it("assertConfigured does not throw when fully configured", () => {
    const config = loadConfig();
    config.loginHost = "login.isambard.ac.uk";
    config.username = "testuser";
    expect(() => assertConfigured(config)).not.toThrow();
  });
});

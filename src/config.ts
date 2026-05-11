import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";

export interface Config {
  loginHost: string;
  username: string;
  projectDir: string;
  defaultLocalPort: number;
  hfToken?: string;
}

const CONFIG_DIR = join(homedir(), ".config", "ivllm");
const CONFIG_PATH = join(CONFIG_DIR, "config.json");

const DEFAULTS: Config = {
  loginHost: "",
  username: "",
  projectDir: "$PROJECTDIR",
  defaultLocalPort: 11434,
};

export function loadConfig(): Config {
  if (!existsSync(CONFIG_PATH)) {
    return { ...DEFAULTS };
  }
  const raw = readFileSync(CONFIG_PATH, "utf-8");
  return { ...DEFAULTS, ...JSON.parse(raw) } as Config;
}

export function saveConfig(config: Config): void {
  if (!existsSync(CONFIG_DIR)) {
    mkdirSync(CONFIG_DIR, { recursive: true });
  }
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2) + "\n", "utf-8");
}

export function assertConfigured(config: Config): void {
  if (!config.loginHost) {
    throw new Error("loginHost not configured. Run: ivllm config --login-host <host>");
  }
  if (!config.username) {
    throw new Error("username not configured. Run: ivllm config --username <user>");
  }
}

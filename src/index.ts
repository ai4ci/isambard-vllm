#!/usr/bin/env bun
import { cmdSetup } from "./commands/setup.ts";
import { cmdStart } from "./commands/start.ts";
import { cmdStatus } from "./commands/status.ts";
import { cmdStop } from "./commands/stop.ts";
import { loadConfig, saveConfig } from "./config.ts";

const [, , command, ...args] = process.argv;

const USAGE = `
Usage: ivllm <command> [options]

Commands:
  setup                   Install vLLM on the HPC (one-off)
  start <job>             Start an inference session and monitor it
  status [job]            Show status of a job (or all jobs)
  stop <job>              Stop a job and clean up (recovery)
  config                  Show or set configuration

Run 'ivllm <command> --help' for command-specific options.
`.trim();

switch (command) {
  case "setup":
    await cmdSetup(args);
    break;
  case "start":
    await cmdStart(args);
    break;
  case "status":
    await cmdStatus(args);
    break;
  case "stop":
    await cmdStop(args);
    break;
  case "config":
    await cmdConfig(args);
    break;
  default:
    console.log(USAGE);
    process.exit(command ? 1 : 0);
}

async function cmdConfig(args: string[]): Promise<void> {
  const config = loadConfig();
  const flags: Record<string, string> = {};
  for (let i = 0; i < args.length - 1; i++) {
    if (args[i]?.startsWith("--")) flags[args[i]!.slice(2)] = args[i + 1] ?? "";
  }
  if (Object.keys(flags).length === 0) {
    console.log(JSON.stringify(config, null, 2));
    return;
  }
  if (flags["login-host"]) config.loginHost = flags["login-host"]!;
  if (flags["username"]) config.username = flags["username"]!;
  if (flags["project-dir"]) config.projectDir = flags["project-dir"]!;
  if (flags["default-local-port"]) config.defaultLocalPort = parseInt(flags["default-local-port"]!, 10);
  if (flags["vllm-version"]) config.vllmVersion = flags["vllm-version"]!;
  saveConfig(config);
  console.log("Configuration saved.");
}

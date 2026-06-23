#!/usr/bin/env bun
import { cmdSetup } from "./commands/setup.ts";
import { cmdStart } from "./commands/start.ts";
import { cmdStatus } from "./commands/status.ts";
import { cmdStop } from "./commands/stop.ts";
import { cmdAgent } from "./commands/agent.ts";
import { loadConfig, saveConfig } from "./config.ts";

const { version } = await import("../package.json");

const [, , command, ...args] = process.argv;

const USAGE = `
Usage: ivllm <command> [options]

Commands:
  setup <version>         Install vLLM <version> on the HPC (one-off, e.g. ivllm setup 0.19.1)
  start <job>             Start an inference session and monitor it
  status [job]            Show status of a job (or all jobs)
  stop <job>              Stop a job and clean up (recovery)
  config                  Show or set configuration
  agent                   Launch AI assistant connected to local vLLM server (interactive menu)

Options:
  --version, -v           Show version

Run 'ivllm <command> --help' for command-specific options.

For command-specific help, run:
  ivllm start --help      Start options (including --no-launch)
  ivllm setup --help      Setup options
  ivllm agent --help      Agent options (including --port)
  ivllm config --help     Config options
`.trim();

switch (command) {
  case "--version":
  case "-v":
    console.log(`ivllm ${version}`);
    process.exit(0);
    break;
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
  case "agent":
    await cmdAgent(args);
    break;
  default:
    console.log(USAGE);
    process.exit(command ? 1 : 0);
}

async function cmdConfig(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
Usage: ivllm config [options]

Options:
  --login-host <host>     SSH login node (e.g. XXXX.aip2.isambard)
  --username <user>       HPC username (e.g. YYYY.XXXX)
  --project-dir <path>    HPC project dir (e.g. /projects/XXXX)
  --default-local-port <port>  Local port for API (default: 11434)
  --hf-token <token>      HuggingFace token for gated models
  --help, -h              Show this help message

Examples:
  ivllm config --login-host XXXX.aip2.isambard --username YYYY.XXXX --project-dir /projects/XXXX
  ivllm config --hf-token hf_...
  ivllm config  # Show current configuration
`);
    return;
  }

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
  if (flags["hf-token"]) config.hfToken = flags["hf-token"]!;
  saveConfig(config);
  console.log("Configuration saved.");
}

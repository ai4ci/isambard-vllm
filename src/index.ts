#!/usr/bin/env bun
import { cmdSetup } from './commands/setup.ts';
import { cmdStart } from './commands/start.ts';
import { cmdStatus } from './commands/status.ts';
import { cmdStop } from './commands/stop.ts';
import { cmdList } from './commands/list.ts';
import { cmdInteractive } from './commands/interactive.ts';
import { cmdAgent } from './commands/agent.ts';
import { cmdConfig } from './commands/config.ts';

// Assign globally across Node.js/Browser using the universal globalThis object
const { version } = await import('../package.json');
(globalThis as any).__VERSION__ = version;

const [, , command, ...args] = process.argv;

const USAGE = `
Usage: ivllm <command> [options]

Commands:
  setup <version>         Install vLLM <version> on the HPC (one-off, e.g. ivllm setup 0.19.1)
  start <job>             Start an inference session and monitor it
  interactive <job>       Start an interactive inference session (bound to terminal)
  list                    List stored vLLM job configs
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
  case '--version':
  case '-v':
    console.log(`ivllm ${__VERSION__}`);
    process.exit(0);
  case 'setup':
    await cmdSetup(args);
    break;
  case 'start':
    await cmdStart(args);
    break;
  case 'status':
    await cmdStatus(args);
    break;
  case 'stop':
    await cmdStop(args);
    break;
  case 'interactive':
    await cmdInteractive(args);
    break;
  case 'list':
    await cmdList(args);
    break;
  case 'config':
    await cmdConfig(args);
    break;
  case 'agent':
    await cmdAgent(args);
    break;
  default:
    console.log(USAGE);
    process.exit(command ? 1 : 0);
}

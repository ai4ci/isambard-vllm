import { mkdtempSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { loadConfig, assertConfigured } from '../config.ts';
import { makeRemoteOps } from '../remote-ops.ts';
import { parseStartArgs } from '../job.ts';
import { runInferenceSession } from '../session-helper.ts';

/**
 *
 * @param args
 */
export async function cmdStart(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: ivllm start <job> [options]

Options:
  --config <file>       vLLM config YAML (contains model, parallelism and all serving options)
  --local-port <n>      Local port to expose the API on (from ivllm config)
  --gpus <n>            GPUs to request (overrides tensor-parallel-size × pipeline-parallel-size from YAML)
  --time <hh:mm:ss>     SLURM time limit (default: 4:00:00)
  --mock                Use mock vLLM server (no GPU needed -- for testing); requires --model
  --dry-run             Preview generated scripts and scp commands without running anything
  --no-launch           Skip assistant launch menu, show config snippet only
  --help, -h            Show this help message

Examples:
  ivllm start my-job --config examples/qwen2.5-instruct.yaml
  ivllm start test-job --mock --model Qwen/Qwen2.5-0.5B-Instruct --dry-run
`);
    return;
  }

  const config = loadConfig();
  try {
    assertConfigured(config);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  let startArgs;
  try {
    startArgs = parseStartArgs(args);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    console.error(
      'Usage: ivllm start <job> [--config <file>] [--local-port <port>] [--gpus <n>] [--time <hh:mm:ss>]',
    );
    console.error(
      '       ivllm start <job> --mock --model <model> [--local-port <port>] [--time <hh:mm:ss>]',
    );
    process.exit(1);
  }

  const dryRunDir = startArgs.dryRun
    ? mkdtempSync(join(tmpdir(), 'ivllm-dryrun-'))
    : undefined;
  const ops = makeRemoteOps(config, startArgs.dryRun, dryRunDir);

  // Delegate to unified session pipeline
  await runInferenceSession(config, startArgs, false, ops);
}

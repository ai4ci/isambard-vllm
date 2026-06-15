import type { Config } from './config.ts';
import { runRemote } from './ssh.ts';
import type { InferenceScriptOptions } from './templates/inference.ts';
// import type { EnvVarEntry } from './vllm-config.ts';

export type JobState = 'running' | 'completed' | 'failed';
export type SlurmQueueState = { state: string; reason: string };
export const SACCT_DIAGNOSTICS_FORMAT =
  'JobID,JobName%24,NodeList%24,State,ExitCode,ReqMem,AllocTRES%40,MaxRSS,MaxRSSNode%18,MaxRSSTask,MaxVMSize';

export function buildSacctDiagnosticsCommand(jobId: string): string {
  return `sacct -j ${jobId} --format=${SACCT_DIAGNOSTICS_FORMAT}`;
}

export function sacctDiagnosticsSettled(
  sacctOutput: string,
  jobId: string,
): boolean {
  const jobLine = sacctOutput
    .split('\n')
    .map((line) => line.trim())
    .find((line) => line.startsWith(`${jobId} `));

  if (!jobLine) return false;
  return !/\b(RUNNING|PENDING|CONFIGURING|COMPLETING)\b/i.test(jobLine);
}

export function parseJobId(sbatchOutput: string): string | null {
  const match = sbatchOutput.match(/Submitted batch job (\d+)/);
  return match?.[1] ?? null;
}

export function parseJobState(sacctOutput: string): JobState | null {
  const state = sacctOutput.trim().split(/\s+/)[0]?.toUpperCase();
  if (!state) return null;
  if (state === 'COMPLETED') return 'completed';
  if (state === 'RUNNING' || state === 'PENDING') return 'running';
  return 'failed';
}

/**
 * Run a job using sbatch. This will place it in the queue. The script must include all the node requirements
 * In the #SBATCH directives at the beginning of the script
 */
export async function submitJob(
  config: Config,
  remoteScriptPath: string,
): Promise<string> {
  const { stdout, exitCode } = await runRemote(
    config,
    `sbatch ${remoteScriptPath}`,
    { silent: true },
  );
  if (exitCode !== 0)
    throw new Error(`sbatch failed (exit ${exitCode}): ${stdout}`);
  const jobId = parseJobId(stdout);
  if (!jobId)
    throw new Error(`Could not parse job ID from sbatch output: ${stdout}`);
  return jobId;
}

/**
 * Run interactively using srun. This will stil be queued but can use the section reserved for interactive.
 * The command must include all the node requirements as CLI flags.
 * #SBATCH directives at the beginning of the script are ignored.
 */
export async function runInteractive(
  config: Config,
  options: InferenceScriptOptions,
  remoteScriptPath: string,
): Promise<string> {
  // Calculate basic metrics
  const gpusPerNode = Math.floor(options.gpuCount / options.nodeCount);
  const isFullOrMultiNode = options.nodeCount > 1 || options.gpuCount === 4;

  // 1. Calculate --mem
  // Use '0' for full/multi node, otherwise scale linearly (120GB per requested GPU)
  const mem = isFullOrMultiNode ? '0' : `${options.gpuCount * 120}G`;

  // 2. Calculate --cpus-per-task
  // Use 64 for full/multi node. For fractional nodes, give 64 cores per GPU to accelerate loading
  const cpusPerTask = isFullOrMultiNode ? '256' : `${options.gpuCount * 64}`;

  // 3. Add --exclusive if we are claiming the entire node or multiple nodes
  const exclusiveFlag = isFullOrMultiNode ? '--exclusive ' : '';

  // Assemble the final command
  const cmd = `srun --nodes=${options.nodeCount} --gpus-per-node=${gpusPerNode} --cpus-per-task=${cpusPerTask} --mem=${mem} ${exclusiveFlag}--time=${options.timeLimit} bash ${remoteScriptPath}`;

  const { stdout, exitCode } = await runRemote(config, cmd, { silent: false });

  if (exitCode !== 0)
    throw new Error(`srun failed (exit ${exitCode}): ${stdout}`);
  const jobId = parseJobId(stdout);
  if (!jobId)
    throw new Error(`Could not parse job ID from sbatch output: ${stdout}`);
  return jobId;
}

export async function pollJobStatus(
  config: Config,
  jobId: string,
): Promise<JobState> {
  const { stdout } = await runRemote(
    config,
    `sacct -j ${jobId} --format=State --noheader -X`,
    { silent: true },
  );
  return parseJobState(stdout) ?? 'running';
}

export async function getJobLog(
  config: Config,
  logPath: string,
): Promise<string> {
  const { stdout } = await runRemote(config, `cat ${logPath}`, {
    silent: true,
  });
  return stdout;
}

export function parseSlurmQueueState(
  squeueOutput: string,
): SlurmQueueState | null {
  const line = squeueOutput.trim();
  if (!line) return null;
  const parts = line.split(/\s+/);
  const state = parts[0] ?? '';
  const reason = parts.slice(1).join(' ') || '';
  return state ? { state, reason } : null;
}

export async function getSlurmQueueState(
  config: Config,
  jobId: string,
): Promise<SlurmQueueState | null> {
  const { stdout } = await runRemote(
    config,
    `squeue -j ${jobId} --format="%T %R" --noheader 2>/dev/null`,
    { silent: true },
  );
  return parseSlurmQueueState(stdout);
}

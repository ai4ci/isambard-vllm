import type { Config } from './config.ts';
import { runRemote, streamSrun } from './ssh.ts';
import type { InferenceScriptOptions } from './templates/inference.ts';
import type { SessionState, MonitorRuntimeOpts } from './session-helper.ts';
import type { StartArgs } from './job.ts';

export type JobState = 'running' | 'completed' | 'failed';
export type SlurmQueueState = { state: string; reason: string };
export const SACCT_DIAGNOSTICS_FORMAT =
  'JobID,JobName%24,NodeList%24,State,ExitCode,ReqMem,AllocTRES%40,MaxRSS,MaxRSSNode%18,MaxRSSTask,MaxVMSize';

/**
 *
 * @param jobId
 */
export function buildSacctDiagnosticsCommand(jobId: string): string {
  return `sacct -j ${jobId} --format=${SACCT_DIAGNOSTICS_FORMAT}`;
}

/**
 *
 * @param sacctOutput
 * @param jobId
 */
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

/**
 *
 * @param sbatchOutput
 */
export function parseJobId(sbatchOutput: string): string | null {
  const match = sbatchOutput.match(/Submitted batch job (\d+)/);
  return match?.[1] ?? null;
}

/**
 *
 * @param sacctOutput
 */
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
 * @param config
 * @param remoteScriptPath
 */
export async function submitJob(
  config: Config,
  remoteScriptPath: string,
  sessionState: SessionState,
  startArgs: StartArgs,
  opts: MonitorRuntimeOpts,
  monitor: (
    state: SessionState,
    args: StartArgs,
    runtimeOpts: MonitorRuntimeOpts,
  ) => Promise<void>,
): Promise<void> {
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

  console.log(`✓ SLURM job submitted: ${jobId}`);
  sessionState.slurmJobId = jobId;

  // Sbatch returns immediately, so block/await monitorSession here
  await monitor(sessionState, startArgs, opts);
}

/**
 * Run interactively using srun. This will stil be queued but can use the section reserved for interactive.
 * The command must include all the node requirements as CLI flags.
 * #SBATCH directives at the beginning of the script are ignored.
 * @param config
 * @param options
 * @param remoteScriptPath
 */
export async function runInteractive(
  config: Config,
  options: InferenceScriptOptions,
  remoteScriptPath: string,
  sessionState: SessionState,
  startArgs: StartArgs,
  opts: MonitorRuntimeOpts,
  monitor: (
    state: SessionState,
    args: StartArgs,
    runtimeOpts: MonitorRuntimeOpts,
  ) => Promise<void>,
): Promise<void> {
  // Calculate basic metrics
  const gpusPerNode = Math.floor(options.gpuCount / options.nodeCount);
  const isFullOrMultiNode = options.nodeCount > 1 || options.gpuCount === 4;

  // 1. Calculate --mem
  // Use '0' for full/multi node, otherwise scale linearly (120GB per GPU, one
  // Grace Hopper superchip per GPU).
  const mem = '0'; // isFullOrMultiNode ? '0' : `${options.gpuCount * 120}G`;

  // 2. Calculate --cpus-per-task
  // Each Grace Hopper superchip has 72 physical cores + 1 H100 GPU. We use 64
  // cores per GPU for fractional nodes (leaving ~8 for model-loading/IO).
  // Full and multi-node get all 256 threads (4 × 64) since the node is exclusive.
  const cpusPerTask = isFullOrMultiNode ? '256' : `${options.gpuCount * 64}`;

  // 3. Add --exclusive if we are claiming the entire node or multiple nodes
  const exclusiveFlag = isFullOrMultiNode ? '--exclusive ' : '';

  // Assemble the final command
  const cmd = `srun --nodes=${options.nodeCount} --gpus-per-node=${gpusPerNode} --cpus-per-task=${cpusPerTask} --mem=${mem} ${exclusiveFlag}--time=${options.timeLimit} bash ${remoteScriptPath}`;

  console.log(`Executing: ${cmd}`);

  const { exitCode } = await streamSrun(config, cmd, {
    silent: false,
    onIdReceived: (jobId) => {
      console.log(`\n🚀 Captured Slurm/Srun Identifier: ${jobId}`);
      sessionState.slurmJobId = jobId;

      // Start monitoring asynchronously in the background as srun logs stream
      monitor(sessionState, startArgs, opts).catch((err) => {
        console.error('Error in monitor session:', err);
      });
    },
  });

  if (exitCode !== 0) {
    throw new Error(`srun failed (exit ${exitCode})`);
  }
}

/**
 *
 * @param config
 * @param jobId
 */
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

/**
 *
 * @param config
 * @param logPath
 */
export async function getJobLog(
  config: Config,
  logPath: string,
): Promise<string> {
  const { stdout } = await runRemote(config, `cat ${logPath}`, {
    silent: true,
  });
  return stdout;
}

/**
 *
 * @param squeueOutput
 */
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

/**
 *
 * @param config
 * @param jobId
 */
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

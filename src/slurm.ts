import type {
  SessionState,
  RemoteMonitor,
  RemoteOps,
  ProcessState,
} from './types.ts';

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
  remoteScriptPath: string,
  sessionState: ProcessState,
  monitor?: RemoteMonitor,
): Promise<void> {
  const ops = sessionState.ops;
  const { stdout, exitCode } = await ops.runRemote(
    `sbatch ${remoteScriptPath}`,
  );
  if (exitCode !== 0)
    throw new Error(`sbatch failed (exit ${exitCode}): ${stdout}`);

  const jobId = parseJobId(stdout);
  if (!jobId)
    throw new Error(`Could not parse job ID from sbatch output: ${stdout}`);

  console.log(`✓ SLURM job submitted: ${jobId}`);
  sessionState.slurmJobId = jobId;

  // Sbatch returns immediately, so no process to manage
  // monitor may detach or keep alive.
  if (monitor) await monitor.start(sessionState);
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
  ops: RemoteOps,
  remoteScriptPath: string,
  sessionState: SessionState,
  monitor: RemoteMonitor,
): Promise<void> {
  const options = sessionState.startArgs!;
  const nodeCount = Math.ceil(options.gpuCount / 4);
  // Calculate basic metrics
  const gpusPerNode = Math.ceil(options.gpuCount / nodeCount);
  const isFullOrMultiNode = nodeCount > 1 || options.gpuCount === 4;

  // 1. Calculate --mem
  // Use '0' for full/multi node, otherwise scale linearly (115GB usable per GPU, one
  // Grace Hopper superchip per GPU).
  const mem = isFullOrMultiNode ? '0' : `${options.gpuCount * 115}G`;

  // 2. Calculate --cpus-per-task
  // Each Grace Hopper superchip has 72 physical cores + 1 H100 GPU. We use 64
  // cores per GPU for fractional nodes (leaving ~8 for model-loading/IO).
  // Full and multi-node get all 256 threads (4 × 64) since the node is exclusive.
  // const cpusPerTask = isFullOrMultiNode ? '256' : `${options.gpuCount * 64}`;

  // 3. Add --exclusive if we are claiming the entire node or multiple nodes
  const exclusiveFlag = isFullOrMultiNode ? '--exclusive ' : '';

  //TODO: consider interactive and --export=ALL?
  const cmd = `srun --job-name=${options.jobName} --nodes=${nodeCount} --gpus-per-node=${gpusPerNode} --cpus-per-gpu=64 --cpu-bind=cores --ntasks-per-node=1 --partition=interactive --reservation=interactive --mem=${mem} ${exclusiveFlag}--time=${options.timeLimit} bash ${remoteScriptPath}`;

  console.log(`Executing: ${cmd}`);

  sessionState.process = ops.streamSrun(cmd, sessionState, {
    env: [],
    silent: false,
  });
  monitor.start(sessionState);
}

/**
 *
 * @param config
 * @param jobId
 */
export async function pollJobStatus(
  ops: RemoteOps,
  jobId: string,
): Promise<JobState> {
  const { stdout } = await ops.runRemote(
    `sacct -j ${jobId} --format=State --noheader -X`,
    { env: [], silent: true },
  );
  return parseJobState(stdout) ?? 'running';
}

/**
 *
 * @param config
 * @param logPath
 */
export async function getJobLog(
  ops: RemoteOps,
  logPath: string,
): Promise<string> {
  const { stdout } = await ops.runRemote(`cat ${logPath}`, {
    env: [],
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
  ops: RemoteOps,
  jobId: string,
): Promise<SlurmQueueState | null> {
  const { stdout } = await ops.runRemote(
    `squeue -j ${jobId} --format="%T %R" --noheader 2>/dev/null`,
    { env: [], silent: true },
  );
  return parseSlurmQueueState(stdout);
}

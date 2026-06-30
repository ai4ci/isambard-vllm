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
 * Build a sacct diagnostics command for post-mortem analysis.
 *
 * Uses the {@link SACCT_DIAGNOSTICS_FORMAT} to request job details
 * including exit code, resource usage, and memory peaks.
 * @param jobId - The SLURM job ID (e.g. '123456')
 * @returns The full sacct command string
 */
export function buildSacctDiagnosticsCommand(jobId: string): string {
  return `sacct -j ${jobId} --format=${SACCT_DIAGNOSTICS_FORMAT}`;
}

/**
 * Check whether a SLURM job has settled to a terminal state.
 *
 * Parses the sacct output looking for the given job ID. A job is
 * considered settled when its state line does not contain any active
 * state keywords (RUNNING, PENDING, CONFIGURING, COMPLETING).
 * @param sacctOutput - Raw output from a sacct command
 * @param jobId - The SLURM job ID to check for
 * @returns true when the job is no longer active, false if still running
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
 * Extract the SLURM batch job ID from `sbatch` output.
 *
 * Parses the standard sbatch message `Submitted batch job <ID>` and
 * returns the numeric job ID, or `null` if no match is found.
 * @param sbatchOutput - Raw stdout from an sbatch command
 * @returns The job ID string, or `null` if parsing fails
 */
export function parseJobId(sbatchOutput: string): string | null {
  const match = sbatchOutput.match(/Submitted batch job (\d+)/);
  return match?.[1] ?? null;
}

/**
 * Parse a SLURM job state from `sacct` output and map it to a
 * {@link JobState} enum value.
 *
 * Reads only the first whitespace-delimited token of the state column:
 *
 * | Token     | Mapped state  |
 * |-----------|---------------|
 * | `COMPLETED` | `completed` |
 * | `RUNNING` / `PENDING` | `running` |
 * | Anything else | `failed` |
 * @param sacctOutput - Raw state string from `sacct --format=State`
 * @returns A mapped job state, or `null` for empty input
 */
export function parseJobState(sacctOutput: string): JobState | null {
  const state = sacctOutput.trim().split(/\s+/)[0]?.toUpperCase();
  if (!state) return null;
  if (state === 'COMPLETED') return 'completed';
  if (state === 'RUNNING' || state === 'PENDING') return 'running';
  return 'failed';
}

/**
 * Submit a SLURM batch job via `sbatch` and start an optional monitor.
 *
 * Runs the specified script on the remote host, extracts the job ID
 * from sbatch output, and stores it in the {@link ProcessState}. After
 * submission the optional {@link monitor} is started to observe progress.
 * @param remoteScriptPath - Path to the SLURM script on the remote host
 * @param sessionState - Process state whose `slurmJobId` is updated on success
 * @param monitor - Optional monitor to observe the job after submission
 * @throws {Error} if sbatch fails or the job ID cannot be parsed
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
 * Submit a job script for interactive execution via `srun`.
 *
 * Calculates resource parameters (GPU count, CPUs per GPU, memory, node
 * exclusivity) from the session's start arguments and executes the remote
 * script through an {@link RemoteOps.streamSrun} stream. Unlike
 * {@link submitJob}, this keeps the terminal coupled to job output.
 *
 * **Resource calculation**:
 *
 * - **Node count**: `ceil(gpuCount / 4)`
 * - **Gpus per node**: `ceil(gpuCount / nodeCount)`
 * - **Memory**: `'0'` (unlimited) for full/multi-node; otherwise
 *   `gpuCount × 115 GB` (one Grace Hopper superchip per GPU)
 * - **CPUs per GPU**: Fixed at 64 (8 physical cores reserved for IO)
 * - **Exclusive flag**: Applied when `nodeCount > 1` or all 4 GPUs used
 * @param ops - {@link RemoteOps} instance for SSH execution
 * @param remoteScriptPath - Absolute path to the remote `.sh` script
 * @param sessionState - Current session (provides `startArgs`, `process`)
 * @param monitor - Monitor to start after the srun process begins
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

  const cmd = `srun --job-name=${options.jobName} --nodes=${nodeCount} --gpus-per-node=${gpusPerNode} --cpus-per-gpu=64 --cpu-bind=cores --ntasks-per-node=1 --partition=interactive --reservation=interactive --mem=${mem} ${exclusiveFlag}--time=${options.timeLimit} bash ${remoteScriptPath}`;

  console.log(`Executing: ${cmd}`);

  sessionState.process = ops.streamSrun(cmd, sessionState, {
    env: [],
    silent: false,
  });
  monitor.start(sessionState);
}

/**
 * Poll a SLURM job's current state via `sacct`.
 *
 * Queries the job state using `sacct --format=State --noheader -X` and
 * delegates to {@link parseJobState} for mapping. Returns `'running'` as
 * a fallback when the state cannot be parsed.
 * @param ops - {@link RemoteOps} instance for SSH execution
 * @param jobId - SLURM job ID to query
 * @returns The job state, defaulting to `'running'` if unparseable
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
 * Fetch the contents of a remote log file via SSH.
 *
 * A simple wrapper around `cat` executed over the login node connection.
 * @param ops - {@link RemoteOps} instance for SSH execution
 * @param logPath - Absolute path to the remote log file
 * @returns The file contents as a string
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
 * Parse a `squeue` output line into a structured {@link SlurmQueueState}.
 *
 * Splits the trimmed output on whitespace — the first token becomes
 * the state, and the rest is joined as the reason.
 * @param squeueOutput - Raw output from `squeue -j <id> --format="%T %R"`
 * @returns Parsed state and reason, or `null` for empty input
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
 * Query the SLURM queue for a job's current state and reason.
 *
 * Runs `squeue -j <jobId> --format="%T %R"` on the login node and
 * delegates to {@link parseSlurmQueueState} for parsing. Returns
 * null when the job is not found in the queue.
 * @param ops - Remote operations interface for SSH execution
 * @param jobId - The SLURM job ID to query
 * @returns The parsed queue state and reason, or null if the job is not in the queue
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

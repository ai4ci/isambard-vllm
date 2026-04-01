import type { Config } from "./config.ts";
import { runRemote } from "./ssh.ts";

export type JobState = "running" | "completed" | "failed";

export function parseJobId(sbatchOutput: string): string | null {
  const match = sbatchOutput.match(/Submitted batch job (\d+)/);
  return match?.[1] ?? null;
}

export function parseJobState(sacctOutput: string): JobState | null {
  const state = sacctOutput.trim().split(/\s+/)[0]?.toUpperCase();
  if (!state) return null;
  if (state === "COMPLETED") return "completed";
  if (state === "RUNNING" || state === "PENDING") return "running";
  return "failed";
}

export async function submitJob(config: Config, remoteScriptPath: string): Promise<string> {
  const { stdout, exitCode } = await runRemote(config, `sbatch ${remoteScriptPath}`, { silent: true });
  if (exitCode !== 0) throw new Error(`sbatch failed (exit ${exitCode}): ${stdout}`);
  const jobId = parseJobId(stdout);
  if (!jobId) throw new Error(`Could not parse job ID from sbatch output: ${stdout}`);
  return jobId;
}

export async function pollJobStatus(config: Config, jobId: string): Promise<JobState> {
  const { stdout } = await runRemote(
    config,
    `sacct -j ${jobId} --format=State --noheader -X`,
    { silent: true }
  );
  return parseJobState(stdout) ?? "running";
}

export async function getJobLog(config: Config, logPath: string): Promise<string> {
  const { stdout } = await runRemote(config, `cat ${logPath}`, { silent: true });
  return stdout;
}

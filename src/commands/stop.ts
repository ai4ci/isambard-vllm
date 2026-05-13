import { spawn } from "child_process";
import { loadConfig, assertConfigured } from "../config.ts";
import { runRemote } from "../ssh.ts";
import { parseJobDetails } from "../job.ts";

export interface StopArgs {
  jobName: string;
}

export function parseStopArgs(args: string[]): StopArgs {
  const jobName = args[0] && !args[0].startsWith("--") ? args[0] : null;
  if (!jobName) throw new Error("Job name is required as the first argument");
  return { jobName };
}

export async function cmdStop(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
Usage: ivllm stop <job>

Options:
  <job>                 Name of the job to stop and clean up
  --help, -h            Show this help message

Examples:
  ivllm stop my-job
`);
    return;
  }

  const config = loadConfig();
  try { assertConfigured(config); } catch (e) { console.error("Error:", (e as Error).message); process.exit(1); }

  let stopArgs: StopArgs;
  try {
    stopArgs = parseStopArgs(args);
  } catch (e) {
    console.error("Error:", (e as Error).message);
    console.error("Usage: ivllm stop <job>");
    process.exit(1);
  }

  const { jobName } = stopArgs;
  const remoteJobDetails = `~/${jobName}/job_details.json`;

  console.log(`=== ivllm stop: ${jobName} ===`);

  // Read job_details.json to get SLURM job ID
  const { stdout } = await runRemote(config, `cat ${remoteJobDetails} 2>/dev/null`, { silent: true });
  const details = parseJobDetails(stdout);

  if (!details) {
    console.error(`No job '${jobName}' found (no job_details.json). Nothing to clean up.`);
    process.exit(1);
  }

  // Cancel SLURM job if we have an ID
  if (details.slurm_job_id) {
    process.stdout.write(`  Cancelling SLURM job ${details.slurm_job_id}...`);
    const { exitCode } = await runRemote(
      config, `scancel ${details.slurm_job_id}`, { silent: true }
    );
    console.log(exitCode === 0 ? " done" : " (scancel returned non-zero, job may have already ended)");
  } else {
    console.log("  No SLURM job ID in job_details — skipping scancel");
  }

  // Remove lockfile on HPC
  process.stdout.write("  Removing lockfile on HPC...");
  await runRemote(config, `rm -f ${remoteJobDetails}`, { silent: true });
  console.log(" done");

  // Best-effort: terminate any local orphaned tunnel for the default port
  const localPort = config.defaultLocalPort ?? 11434;
  await cleanupLocalTunnel(localPort);

  console.log(`✓ Job '${jobName}' stopped and cleaned up.`);
}

/** Attempt to terminate any local SSH forward-tunnel processes for the given port. */
function cleanupLocalTunnel(localPort: number): Promise<void> {
  return new Promise(resolve => {
    const pattern = `ssh.*-L.*${localPort}:`;
    const proc = spawn("pkill", ["-f", pattern], { stdio: "ignore" });
    proc.on("close", (code) => {
      if (code === 0) console.log(`  Terminated local SSH tunnel on port ${localPort}`);
      // exit code 1 means no process matched — that's fine
      resolve();
    });
    proc.on("error", () => resolve()); // pkill not available — ignore
  });
}

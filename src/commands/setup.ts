import { writeFileSync, unlinkSync, existsSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { loadConfig, assertConfigured } from "../config.ts";
import { runRemote, copyFile } from "../ssh.ts";
import { renderSetupScript } from "../templates/setup.ts";
import { submitJob, pollJobStatus, getJobLog, getSlurmQueueState } from "../slurm.ts";

import { tailRemoteLog } from "../ssh.ts";

const POLL_INTERVAL_MS = 60_000; // Isambard policy: do not poll Slurm scheduler more than once per minute
const REMOTE_SCRIPT_PATH = "~/.config/ivllm/setup.slurm.sh";
const REMOTE_LOG_PATH = "~/.config/ivllm/setup.log";

export async function cmdSetup(args: string[]): Promise<void> {
  const vllmVersion = args[0];
  if (!vllmVersion || vllmVersion.startsWith("--")) {
    console.error("Error: vLLM version is required.");
    console.error("Usage: ivllm setup <version>  (e.g. ivllm setup 0.19.1)");
    process.exit(1);
  }
  const config = loadConfig();
  try { assertConfigured(config); } catch (e) { console.error("Error:", (e as Error).message); process.exit(1); }

  console.log("=== ivllm setup ===");
  console.log(`Login node  : ${config.loginHost}`);
  console.log(`vLLM        : ${vllmVersion}`);
  console.log(`Install dir : ${config.projectDir}/ivllm/${vllmVersion}`);
  console.log("");

  // Pre-flight: check SSH connectivity
  console.log("Checking SSH connectivity...");
  const { exitCode: sshCheck } = await runRemote(config, "echo ok", { silent: true });
  if (sshCheck !== 0) {
    console.error("Error: Cannot connect to login node. Check your SSH configuration.");
    process.exit(1);
  }
  console.log("✓ SSH connectivity OK");

  // Check if versioned venv already exists
  const venvDir = `${config.projectDir}/ivllm/${vllmVersion}`;
  const { exitCode: venvCheck } = await runRemote(
    config,
    `test -d ${venvDir}/bin`,
    { silent: true }
  );
  if (venvCheck === 0) {
    console.log(`✓ vLLM ${vllmVersion} already installed at ${venvDir}`);
    console.log("  Delete the directory first to reinstall.");
    return;
  }

  // Render and copy setup script to LOGIN
  const script = renderSetupScript({ vllmVersion });

  const localTmp = join(tmpdir(), "ivllm-setup.slurm.sh");
  writeFileSync(localTmp, script, "utf-8");

  try {
    console.log("Copying setup script to login node...");
    await runRemote(config, `mkdir -p ~/.config/ivllm`, { silent: true });
    await copyFile(config, localTmp, REMOTE_SCRIPT_PATH);
    console.log("✓ Script copied");

    // Submit SLURM job
    console.log("Submitting SLURM setup job...");
    const jobId = await submitJob(config, REMOTE_SCRIPT_PATH);
    console.log(`✓ SLURM job submitted: ${jobId}`);

    // Truncate the log so we only stream this run's output
    await runRemote(config, `truncate -s 0 ${REMOTE_LOG_PATH} 2>/dev/null || true`, { silent: true });

    // Wait for the job to leave the queue (PENDING → RUNNING)
    console.log("Waiting for compute node allocation...");
    while (true) {
      const queueState = await getSlurmQueueState(config, jobId);
      if (!queueState || queueState.state !== "PENDING") break;
      await sleep(POLL_INTERVAL_MS);
    }
    console.log("✓ Job started — streaming setup log (this may take 10–20 minutes)...");
    console.log("─".repeat(60));

    // Stream the remote log to stdout in real time
    // Give the job a moment to start writing the log before tailing
    await sleep(3_000);
    const tail = tailRemoteLog(config, REMOTE_LOG_PATH, "  ");

    // Poll for job completion while log streams in the background
    let state = await pollJobStatus(config, jobId);
    while (state === "running") {
      await sleep(POLL_INTERVAL_MS);
      state = await pollJobStatus(config, jobId);
    }

    // Give tail a moment to flush remaining lines before stopping
    await sleep(2_000);
    tail.stop();
    console.log("─".repeat(60));

    // Fetch log
    const log = await getJobLog(config, REMOTE_LOG_PATH);

    if (state === "failed") {
      console.error("✗ Setup job failed. Log output:");
      console.error(log);
      process.exit(1);
    }

    if (!log.includes("IVLLM_SETUP_SUCCESS")) {
      console.error("✗ Setup job completed but success marker not found. Log output:");
      console.error(log);
      process.exit(1);
    }

    // Validate venv exists
    const { exitCode: finalCheck } = await runRemote(
      config,
      `test -d ${venvDir}/bin`,
      { silent: true }
    );
    if (finalCheck !== 0) {
      console.error(`✗ venv not found at ${venvDir} after setup.`);
      process.exit(1);
    }

    console.log(`✓ vLLM ${vllmVersion} installation complete`);
    const versionLine = log.split("\n").find(l => l.startsWith("vllm"));
    if (versionLine) console.log(`  ${versionLine}`);
  } finally {
    if (existsSync(localTmp)) unlinkSync(localTmp);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

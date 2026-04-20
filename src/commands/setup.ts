import { writeFileSync, unlinkSync, existsSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { loadConfig, assertConfigured } from "../config.ts";
import { runRemote, copyFile } from "../ssh.ts";
import { renderSetupScript } from "../templates/setup.ts";
import { submitJob, pollJobStatus, getJobLog } from "../slurm.ts";

const POLL_INTERVAL_MS = 10_000;
const REMOTE_SCRIPT_PATH = "~/.config/ivllm/setup.slurm.sh";
const REMOTE_LOG_PATH = "~/.config/ivllm/setup.log";

export async function cmdSetup(_args: string[]): Promise<void> {
  const config = loadConfig();
  try { assertConfigured(config); } catch (e) { console.error("Error:", (e as Error).message); process.exit(1); }

  console.log("=== ivllm setup ===");
  console.log(`Login node  : ${config.loginHost}`);
  console.log(`vLLM        : ${config.vllmVersion}`);
  console.log(`Install dir : ${config.projectDir}/ivllm/${config.vllmVersion}`);
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
  const venvDir = `${config.projectDir}/ivllm/${config.vllmVersion}`;
  const { exitCode: venvCheck } = await runRemote(
    config,
    `test -d ${venvDir}/bin`,
    { silent: true }
  );
  if (venvCheck === 0) {
    console.log(`✓ vLLM ${config.vllmVersion} already installed at ${venvDir}`);
    console.log("  Delete the directory first to reinstall.");
    return;
  }

  // Render and copy setup script to LOGIN
  const script = renderSetupScript({
    vllmVersion: config.vllmVersion,
  });

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
    console.log("");
    console.log("Waiting for job to complete (this may take 10–20 minutes)...");

    // Poll for completion
    let state = await pollJobStatus(config, jobId);
    while (state === "running") {
      await sleep(POLL_INTERVAL_MS);
      state = await pollJobStatus(config, jobId);
      process.stdout.write(".");
    }
    console.log("");

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

    console.log(`✓ vLLM ${config.vllmVersion} installation complete`);
    const versionLine = log.split("\n").find(l => l.startsWith("vllm"));
    if (versionLine) console.log(`  ${versionLine}`);
  } finally {
    if (existsSync(localTmp)) unlinkSync(localTmp);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

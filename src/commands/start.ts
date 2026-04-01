import { readFileSync, existsSync } from "fs";
import { writeFileSync, unlinkSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { createInterface } from "readline";
import type { ChildProcess } from "child_process";
import { loadConfig, assertConfigured } from "../config.ts";
import { runRemote, copyFile, spawnTunnel } from "../ssh.ts";
import { submitJob, pollJobStatus } from "../slurm.ts";
import { renderInferenceScript } from "../templates/inference.ts";
import { parseJobDetails, hfCachePath, parseStartArgs, type JobDetails } from "../job.ts";

const HEARTBEAT_INTERVAL_MS = 15_000;
const POLL_INTERVAL_MS = 5_000;

export async function cmdStart(args: string[]): Promise<void> {
  const config = loadConfig();
  try { assertConfigured(config); } catch (e) { console.error("Error:", (e as Error).message); process.exit(1); }

  let startArgs;
  try {
    startArgs = parseStartArgs(args);
  } catch (e) {
    console.error("Error:", (e as Error).message);
    console.error("Usage: ivllm start <job> --model <model> --config <file> [--local-port <port>] [--gpus <n>] [--time <hh:mm:ss>]");
    process.exit(1);
  }

  const { jobName, model, configFile, gpuCount, tensorParallelSize, timeLimit, serverPort } = startArgs;
  const localPort = startArgs.localPort ?? config.defaultLocalPort;
  const hfHome = `${config.projectDir}/hf`;
  const remoteWorkDir = `~/${jobName}`;
  const remoteJobDetails = `${remoteWorkDir}/job_details.json`;

  // Session state for shutdown sequence
  let shuttingDown = false;
  let slurmJobId: string | null = null;
  let tunnel: ChildProcess | null = null;
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null;

  async function shutdown(reason: string, exitCode = 0): Promise<void> {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`\n⏹  Shutting down: ${reason}`);

    if (heartbeatTimer) clearInterval(heartbeatTimer);

    if (slurmJobId) {
      process.stdout.write("  Cancelling SLURM job...");
      await runRemote(config, `scancel ${slurmJobId}`, { silent: true }).catch(() => {});
      console.log(" done");
    }

    if (tunnel) {
      process.stdout.write("  Closing SSH tunnel...");
      tunnel.kill();
      console.log(" done");
    }

    process.stdout.write("  Removing lockfile...");
    await runRemote(config, `rm -f ${remoteJobDetails}`, { silent: true }).catch(() => {});
    console.log(" done");

    console.log("✓ Session ended");
    process.exit(exitCode);
  }

  // Register signal handlers immediately
  process.on("SIGINT", () => shutdown("interrupted (Ctrl+C)"));
  process.on("SIGTERM", () => shutdown("terminated"));

  console.log("=== ivllm start ===");
  console.log(`Job        : ${jobName}`);
  console.log(`Model      : ${model}`);
  console.log(`Config     : ${configFile}`);
  console.log(`GPUs       : ${gpuCount}  |  TP size: ${tensorParallelSize}`);
  console.log(`Local port : ${localPort}  |  Server port: ${serverPort}`);
  console.log("");

  // ── Pre-flight ──────────────────────────────────────────────────────────────
  console.log("Checking SSH connectivity...");
  const { exitCode: sshCheck } = await runRemote(config, "echo ok", { silent: true });
  if (sshCheck !== 0) {
    console.error("Error: Cannot connect to login node.");
    process.exit(1);
  }

  console.log("Checking venv...");
  const { exitCode: venvCheck } = await runRemote(
    config, `test -f ${config.venvPath}/bin/activate`, { silent: true }
  );
  if (venvCheck !== 0) {
    console.error(`Error: vLLM venv not found at ${config.venvPath}. Run 'ivllm setup' first.`);
    process.exit(1);
  }
  console.log("✓ Pre-flight checks passed");

  // ── Model download ───────────────────────────────────────────────────────────
  const cachePath = hfCachePath(hfHome, model);
  const { exitCode: cacheCheck } = await runRemote(
    config, `test -d ${cachePath}`, { silent: true }
  );
  if (cacheCheck === 0) {
    console.log(`✓ Model cached at ${cachePath}`);
  } else {
    console.log(`Downloading model ${model} to ${hfHome} on login node...`);
    const hfToken = process.env["HF_TOKEN"] ?? "";
    const downloadCmd = `source ${config.venvPath}/bin/activate && HF_HOME=${hfHome}${hfToken ? ` HF_TOKEN=${hfToken}` : ""} huggingface-cli download ${model}`;
    const { exitCode: dlCode } = await runRemote(config, downloadCmd);
    if (dlCode !== 0) {
      console.error("Error: Model download failed.");
      process.exit(1);
    }
    console.log("✓ Model downloaded");
  }

  // ── Create lockfile ──────────────────────────────────────────────────────────
  console.log("Creating job working directory and lockfile...");
  await runRemote(config, `mkdir -p ${remoteWorkDir}`, { silent: true });
  const pendingJson = JSON.stringify({ status: "pending", job_name: jobName });
  const { exitCode: lockCode } = await runRemote(
    config,
    // Atomically create only if not existing
    `set -C; echo '${pendingJson}' > ${remoteJobDetails}`,
    { silent: true }
  );
  if (lockCode !== 0) {
    console.error(`Error: Job '${jobName}' already exists (lockfile present). Use 'ivllm stop ${jobName}' to clean up.`);
    process.exit(1);
  }
  console.log("✓ Lockfile created");

  // ── Copy files and submit SLURM job ─────────────────────────────────────────
  const remoteConfigFile = `${remoteWorkDir}/${configFile}`;
  const remoteScriptPath = `${remoteWorkDir}/${jobName}.slurm.sh`;

  const script = renderInferenceScript({
    jobName, model, venvPath: config.venvPath, hfHome,
    configFileName: configFile, workDir: remoteWorkDir,
    serverPort, gpuCount, tensorParallelSize, timeLimit,
  });

  const localScriptTmp = join(tmpdir(), `ivllm-${jobName}.slurm.sh`);
  writeFileSync(localScriptTmp, script, "utf-8");

  try {
    console.log("Copying files to login node...");
    await copyFile(config, configFile, remoteConfigFile);
    await copyFile(config, localScriptTmp, remoteScriptPath);
    console.log("✓ Files copied");

    console.log("Submitting SLURM job...");
    slurmJobId = await submitJob(config, remoteScriptPath);
    console.log(`✓ SLURM job submitted: ${slurmJobId}`);
  } finally {
    if (existsSync(localScriptTmp)) unlinkSync(localScriptTmp);
  }

  // ── Monitor loop ─────────────────────────────────────────────────────────────
  console.log("\nMonitoring job status (Ctrl+C or type 'exit' to stop)...\n");
  let lastStatus = "pending";

  while (!shuttingDown) {
    await sleep(POLL_INTERVAL_MS);
    const { stdout } = await runRemote(config, `cat ${remoteJobDetails} 2>/dev/null`, { silent: true });
    const details = parseJobDetails(stdout);

    if (!details) {
      // File missing — SLURM job may have died without updating it
      const slurmState = await pollJobStatus(config, slurmJobId!);
      if (slurmState === "failed") {
        await shutdown("SLURM job failed unexpectedly", 1);
        return;
      }
      continue;
    }

    if (details.status !== lastStatus) {
      console.log(`  [${timestamp()}] Status: ${details.status}`);
      lastStatus = details.status;
    }

    if (details.status === "failed" || details.status === "timeout") {
      if (details.error) console.error(`  Error: ${details.error}`);
      await printSlurmLog(config, remoteWorkDir, jobName);
      await shutdown(`vLLM ${details.status}`, 1);
      return;
    }

    if (details.status === "running") {
      await onRunning(details);
      return;
    }
  }

  async function onRunning(details: JobDetails): Promise<void> {
    const computeHost = details.compute_hostname!;

    console.log(`\n✓ vLLM is running on ${computeHost}:${serverPort}`);

    // Spawn forward SSH tunnel: LOCAL:localPort -> computeHost:serverPort via LOGIN
    tunnel = spawnTunnel(config, localPort, computeHost, serverPort);
    tunnel.on("exit", (code) => {
      if (!shuttingDown) shutdown(`SSH tunnel exited unexpectedly (code ${code})`, 1);
    });

    // Brief wait for tunnel to establish
    await sleep(2000);

    console.log(`\n🚀 OpenAI API endpoint: http://localhost:${localPort}/v1`);
    console.log(`   Model: ${details.model ?? model}`);
    console.log("\nType 'exit' + Enter to stop, or press Ctrl+C\n");

    // Accept user "exit" command from stdin
    const rl = createInterface({ input: process.stdin, terminal: false });
    rl.on("line", (line) => {
      if (line.trim().toLowerCase() === "exit") {
        rl.close();
        shutdown("user requested exit");
      }
    });

    // Heartbeat loop
    heartbeatTimer = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:${localPort}/health`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
      } catch (e) {
        if (!shuttingDown) {
          console.error(`\nHeartbeat failed: ${(e as Error).message}`);
          await printSlurmLog(config, remoteWorkDir, jobName);
          shutdown("vLLM heartbeat failed", 1);
        }
      }
    }, HEARTBEAT_INTERVAL_MS);
  }
}

async function printSlurmLog(config: Parameters<typeof runRemote>[0], workDir: string, jobName: string): Promise<void> {
  console.error("\n--- SLURM log ---");
  const { stdout } = await runRemote(config, `tail -50 ${workDir}/${jobName}.slurm.log 2>/dev/null`, { silent: true });
  if (stdout) console.error(stdout);
  console.error("--- end log ---\n");
}

function timestamp(): string {
  return new Date().toTimeString().slice(0, 8);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

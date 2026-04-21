import { readFileSync, existsSync } from "fs";
import { writeFileSync, unlinkSync, mkdtempSync } from "fs";
import { join, basename } from "path";
import { tmpdir } from "os";
import { createInterface } from "readline";
import type { ChildProcess } from "child_process";
import { loadConfig, assertConfigured } from "../config.ts";
import { spawnTunnel } from "../ssh.ts";
import { runRemote } from "../ssh.ts";
import { submitJob, pollJobStatus, getSlurmQueueState } from "../slurm.ts";
import { renderInferenceScript } from "../templates/inference.ts";
import { renderMockInferenceScript } from "../templates/mock-inference.ts";
import { parseJobDetails, hfCachePath, parseStartArgs, type JobDetails } from "../job.ts";
import { makeRemoteOps } from "../remote-ops.ts";
import { parseVllmConfig, resolveGpuCount, writeStrippedConfig } from "../vllm-config.ts";
import { semverLt } from "../semver.ts";

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
    console.error("Usage: ivllm start <job> --config <file> [--local-port <port>] [--gpus <n>] [--time <hh:mm:ss>]");
    console.error("       ivllm start <job> --mock --model <model> [--local-port <port>] [--time <hh:mm:ss>]");
    process.exit(1);
  }

  const { jobName, configFile, timeLimit, serverPort } = startArgs;
  const localPort = startArgs.localPort ?? config.defaultLocalPort;
  const hfHome = `${config.projectDir}/hf`;

  // Resolve model and gpuCount: for real mode, read from the vLLM config YAML.
  // For mock mode, model comes from --model CLI flag.
  let model: string;
  let gpuCount: number;
  let nodeCount: number;
  if (startArgs.mock) {
    model = startArgs.model!;
    gpuCount = startArgs.gpuCount ?? 1;
    nodeCount = 1;
  } else {
    let yamlConfig;
    try {
      yamlConfig = parseVllmConfig(configFile!);
    } catch (e) {
      console.error(`Error reading config file '${configFile}': ${(e as Error).message}`);
      process.exit(1);
    }
    if (!yamlConfig.model) {
      console.error(`Error: 'model' is required in the vLLM config file '${configFile}'.`);
      process.exit(1);
    }
    model = yamlConfig.model;
    const resolved = resolveGpuCount(startArgs.gpuCount, yamlConfig);
    gpuCount = resolved.gpuCount;
    nodeCount = resolved.nodeCount;

    // F2.5: enforce min-vllm-version from config
    if (yamlConfig.minVllmVersion) {
      if (semverLt(config.vllmVersion, yamlConfig.minVllmVersion)) {
        console.error(
          `Error: config requires vLLM >= ${yamlConfig.minVllmVersion} but ivllm is configured to use ${config.vllmVersion}.\n` +
          `  Update 'vllm-version' in your ivllm config or use 'ivllm config --vllm-version <version>'.`
        );
        process.exit(1);
      }
    }
  }

  const remoteWorkDir = `$HOME/${jobName}`;     // for SSH commands (remote shell expands $HOME)
  const remoteWorkDirScp = `~/${jobName}`;      // for scp destinations (~ is reliably expanded by scp/sftp; $HOME is not)
  const remoteJobDetails = `${remoteWorkDir}/job_details.json`;

  // Set up dry-run temp dir and RemoteOps
  const dryRunDir = startArgs.dryRun ? mkdtempSync(join(tmpdir(), "ivllm-dryrun-")) : undefined;
  const ops = makeRemoteOps(config, startArgs.dryRun, dryRunDir);

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
      await ops.runRemote(`scancel ${slurmJobId}`, { silent: true }).catch(() => {});
      console.log(" done");
    }

    if (tunnel) {
      process.stdout.write("  Closing SSH tunnel...");
      tunnel.kill();
      console.log(" done");
    }

    process.stdout.write("  Removing lockfile...");
    await ops.runRemote(`rm -f ${remoteJobDetails}`, { silent: true }).catch(() => {});
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
  console.log(`Config     : ${configFile ?? "(N/A — mock mode)"}`);
  console.log(`GPUs       : ${gpuCount}${nodeCount > 1 ? ` (${nodeCount} nodes × ${gpuCount / nodeCount} GPUs each)` : ""}`);
  if (nodeCount > 1) {
    console.log(`⚠ Multi-node job: ${nodeCount} nodes requested`);
  }
  console.log(`Local port : ${localPort}  |  Server port: ${serverPort}`);
  console.log("");

  // ── Pre-flight ──────────────────────────────────────────────────────────────
  if (startArgs.dryRun) {
    console.log("[dry-run] SSH connectivity check skipped");
    console.log("[dry-run] Venv check skipped");
  } else {
    console.log("Checking SSH connectivity...");
    const { exitCode: sshCheck } = await ops.runRemote("echo ok", { silent: true });
    if (sshCheck !== 0) {
      console.error("Error: Cannot connect to login node.");
      process.exit(1);
    }

    console.log("Checking venv...");
    const venvDir = `${config.projectDir}/ivllm/${config.vllmVersion}`;
    const { exitCode: venvCheck } = await ops.runRemote(
      `test -f ${venvDir}/bin/activate`, { silent: true }
    );
    if (venvCheck !== 0) {
      console.error(`Error: vLLM venv not found at ${venvDir}. Run 'ivllm setup' first.`);
      process.exit(1);
    }
    console.log("✓ Pre-flight checks passed");
  }

  // ── Model download ───────────────────────────────────────────────────────────
  const cachePath = hfCachePath(hfHome, model);
  if (startArgs.dryRun) {
    console.log(`[dry-run] HF cache check skipped (would check: ${cachePath})`);
  } else if (startArgs.mock) {
    console.log(`[mock] Model download skipped`);
  } else {
    const { exitCode: cacheCheck } = await ops.runRemote(
      `test -d ${cachePath}`, { silent: true }
    );
    if (cacheCheck === 0) {
      console.log(`✓ Model cached at ${cachePath}`);
    } else {
      console.log(`Downloading model ${model} to ${hfHome} on login node...`);
      const hfToken = process.env["HF_TOKEN"] ?? "";
      const downloadCmd = `source ${config.projectDir}/ivllm/${config.vllmVersion}/bin/activate && HF_HOME=${hfHome}${hfToken ? ` HF_TOKEN=${hfToken}` : ""} hf download ${model}`;
      const { exitCode: dlCode } = await ops.runRemote(downloadCmd);
      if (dlCode !== 0) {
        console.error("Error: Model download failed.");
        process.exit(1);
      }
      console.log("✓ Model downloaded");
    }
  }

  // ── Create lockfile ──────────────────────────────────────────────────────────
  if (startArgs.dryRun) {
    console.log(`[dry-run] Lockfile creation skipped (would create: ${remoteJobDetails})`);
  } else {
    console.log("Creating job working directory and lockfile...");
    await ops.runRemote(`mkdir -p ${remoteWorkDir}`, { silent: true });
    const pendingJson = JSON.stringify({ status: "pending", job_name: jobName });
    const { exitCode: lockCode } = await ops.runRemote(
      `set -C; echo '${pendingJson}' > ${remoteJobDetails}`,
      { silent: true }
    );
    if (lockCode !== 0) {
      console.error(`Error: Job '${jobName}' already exists (lockfile present). Use 'ivllm stop ${jobName}' to clean up.`);
      process.exit(1);
    }
    console.log("✓ Lockfile created");
  }

  // ── Copy files and submit SLURM job ─────────────────────────────────────────
  const remoteConfigFile = configFile ? `${remoteWorkDir}/${basename(configFile)}` : undefined;
  const remoteScriptPath = `${remoteWorkDir}/${jobName}.slurm.sh`;
  // Scp destinations use ~ (reliably expanded by scp/sftp; $HOME is not expanded without a shell)
  const remoteConfigFileScp = configFile ? `${remoteWorkDirScp}/${basename(configFile)}` : undefined;
  const remoteScriptPathScp = `${remoteWorkDirScp}/${jobName}.slurm.sh`;

  const script = startArgs.mock
    ? renderMockInferenceScript({ jobName, model, workDir: remoteWorkDir, serverPort, timeLimit })
    : renderInferenceScript({
        jobName, model, vllmVersion: config.vllmVersion, hfHome,
        configFileName: configFile ? basename(configFile) : "",
        workDir: remoteWorkDir,
        serverPort, gpuCount, nodeCount, timeLimit,
      });

  const localScriptTmp = join(tmpdir(), `ivllm-${jobName}.slurm.sh`);
  writeFileSync(localScriptTmp, script, "utf-8");

  try {
    console.log("Copying files to login node...");
    if (remoteConfigFileScp && configFile) {
      // Strip ivllm-only keys (e.g. min-vllm-version) — vLLM errors on unknown keys
      const strippedConfigTmp = writeStrippedConfig(configFile);
      try {
        await ops.copyFile(strippedConfigTmp, remoteConfigFileScp);
      } finally {
        unlinkSync(strippedConfigTmp);
      }
    }
    await ops.copyFile(localScriptTmp, remoteScriptPathScp);
    console.log("✓ Files copied");

    // ── Dry-run: show summary and exit ────────────────────────────────────────
    if (startArgs.dryRun) {
      console.log(`\n=== Dry-run complete ===`);
      console.log(`Generated files saved to: ${dryRunDir}`);
      if (remoteConfigFile && configFile) {
        console.log(`  ${configFile}  →  ${remoteConfigFile}`);
      }
      console.log(`  ${jobName}.slurm.sh  →  ${remoteScriptPath}`);
      console.log(`\nTo run for real, omit --dry-run.`);
      return;
    }

    console.log("Submitting SLURM job...");
    slurmJobId = await submitJob(config, remoteScriptPath);
    console.log(`✓ SLURM job submitted: ${slurmJobId}`);
  } finally {
    if (existsSync(localScriptTmp)) unlinkSync(localScriptTmp);
  }

  // ── Monitor loop ─────────────────────────────────────────────────────────────
  console.log("\nMonitoring job status (Ctrl+C or type 'exit' to stop)...\n");

  // Issue #1: accept "exit" from stdin immediately (before job is even running)
  const rl = createInterface({ input: process.stdin, terminal: false });
  rl.on("line", (line) => {
    if (line.trim().toLowerCase() === "exit") {
      rl.close();
      shutdown("user requested exit");
    }
  });

  let lastStatus = "pending";
  let lastSlurmQueueState = "";
  let logLineOffset = 0;

  while (!shuttingDown) {
    await sleep(POLL_INTERVAL_MS);
    const { stdout } = await ops.runRemote(`cat ${remoteJobDetails} 2>/dev/null`, { silent: true });
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

    // Issue #2a: while our lockfile shows "pending", report actual SLURM queue state
    if (details.status === "pending") {
      const queueState = await getSlurmQueueState(config, slurmJobId!);
      if (queueState) {
        const msg = queueState.state === "PENDING"
          ? `  [${timestamp()}] Waiting in SLURM queue (${queueState.reason})`
          : `  [${timestamp()}] SLURM state: ${queueState.state}`;
        if (msg !== lastSlurmQueueState) {
          console.log(msg);
          lastSlurmQueueState = msg;
        }
      }
    }

    if (details.status !== lastStatus) {
      if (details.status === "initialising") {
        console.log(`  [${timestamp()}] Job allocated — vLLM is starting up...`);
      } else if (details.status !== "pending") {
        console.log(`  [${timestamp()}] Status: ${details.status}`);
      }
      lastStatus = details.status;
    }

    // Issue #2b: stream SLURM log incrementally during startup
    if (details.status === "initialising") {
      const slurmLogPath = `${remoteWorkDir}/${jobName}.slurm.log`;
      const { stdout: newLines } = await ops.runRemote(
        `tail -n +${logLineOffset + 1} ${slurmLogPath} 2>/dev/null`,
        { silent: true }
      );
      if (newLines.trim()) {
        const lines = newLines.split("\n").filter(l => l.trim());
        for (const line of lines) {
          console.log(`  | ${line}`);
        }
        logLineOffset += lines.length;
      }
    }

    if (details.status === "failed" || details.status === "timeout") {
      if (details.error) console.error(`  Error: ${details.error}`);
      await printSlurmLog(config, remoteWorkDir, jobName);
      await shutdown(`vLLM ${details.status}`, 1);
      return;
    }

    if (details.status === "running") {
      rl.close();
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

    // Re-register "exit" handler on the same readline interface for the running phase
    const rl2 = createInterface({ input: process.stdin, terminal: false });
    rl2.on("line", (line) => {
      if (line.trim().toLowerCase() === "exit") {
        rl2.close();
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

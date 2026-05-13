import { readFileSync, existsSync } from "fs";
import { writeFileSync, unlinkSync, mkdtempSync } from "fs";
import { join, basename, resolve } from "path";
import { tmpdir } from "os";
import { createInterface } from "readline";
import { spawnSync, type ChildProcess } from "child_process";
import { loadConfig, assertConfigured } from "../config.ts";
import { spawnTunnel } from "../ssh.ts";
import { runRemote } from "../ssh.ts";
const { version: ivllmVersion } = await import("../../package.json");
import {
  submitJob,
  pollJobStatus,
  getSlurmQueueState,
  buildSacctDiagnosticsCommand,
  sacctDiagnosticsSettled,
} from "../slurm.ts";
import { renderInferenceScript } from "../templates/inference.ts";
import { renderMockInferenceScript } from "../templates/mock-inference.ts";
import { parseJobDetails, hfCachePath, parseStartArgs, type JobDetails } from "../job.ts";
import { makeRemoteOps } from "../remote-ops.ts";
import { parseVllmConfig, resolveGpuCount, writeStrippedConfig, jobConfigPath, saveJobConfig } from "../vllm-config.ts";
import { semverGte, semverSort } from "../semver.ts";
import { formatOpencodeSnippet } from "../opencode.ts";
import {
  ASSISTANTS,
  type AssistantName,
  type LaunchWrapper,
  getAvailableAssistants,
  getScoderAvailable,
  getSbxAvailable,
  getAssistantLabel,
  getAvailableWrappers,
  buildLaunchCommand,
  buildSandboxName,
  buildSandboxCreateCommand,
  listSbxSandboxes,
  findMatchingSandbox,
  ensureSbxSandbox,
} from "../assistant.ts";

const HEARTBEAT_INTERVAL_MS = 15_000;
const POLL_INTERVAL_MS = 5_000;
// Isambard policy: do not poll Slurm scheduler (squeue/sacct) more than once per minute
const SLURM_POLL_INTERVAL_MS = 60_000;

/**
 * Given a list of installed vLLM versions and a minimum required version,
 * returns the highest installed version that satisfies the minimum, or null if none do.
 */
export function selectBestVersion(installed: string[], minVersion: string): string | null {
  const candidates = installed.filter(v => semverGte(v, minVersion));
  if (candidates.length === 0) return null;
  return semverSort(candidates)[0]!;
}

/**
 * Lists installed vLLM versions by scanning $PROJECTDIR/ivllm/ for versioned venv directories.
 * Returns versions that have a bin/ directory (i.e. are complete installs).
 */
async function listInstalledVersions(config: import("../config.ts").Config, ops: ReturnType<typeof makeRemoteOps>): Promise<string[]> {
  const { stdout } = await ops.runRemote(
    `ls -d ${config.projectDir}/ivllm/*/bin 2>/dev/null | sed 's|.*/ivllm/||; s|/bin||'`,
    { silent: true }
  );
  return stdout.trim().split("\n").filter(v => v && /^\d+\.\d+/.test(v));
}

export async function cmdStart(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes("--help") || args.includes("-h")) {
    console.log(`
Usage: ivllm start <job> [options]

Options:
  --config <file>       vLLM config YAML (contains model, parallelism and all serving options)
  --local-port <n>      Local port to expose the API on (from ivllm config)
  --gpus <n>            GPUs to request (overrides tensor-parallel-size × pipeline-parallel-size from YAML)
  --time <hh:mm:ss>     SLURM time limit (default: 4:00:00)
  --mock                Use mock vLLM server (no GPU needed -- for testing); requires --model
  --dry-run             Preview generated scripts and scp commands without running anything
  --no-launch           Skip assistant launch menu, show config snippet only
  --help, -h            Show this help message

Examples:
  ivllm start my-job --config examples/qwen2.5-instruct.yaml
  ivllm start test-job --mock --model Qwen/Qwen2.5-0.5B-Instruct --dry-run
`);
    return;
  }

  const config = loadConfig();
  try { assertConfigured(config); } catch (e) { console.error("Error:", (e as Error).message); process.exit(1); }

  let startArgs;
  try {
    startArgs = parseStartArgs(args);
  } catch (e) {
    console.error("Error:", (e as Error).message);
    console.error("Usage: ivllm start <job> [--config <file>] [--local-port <port>] [--gpus <n>] [--time <hh:mm:ss>]");
    console.error("       ivllm start <job> --mock --model <model> [--local-port <port>] [--time <hh:mm:ss>]");
    process.exit(1);
  }

  const { jobName, timeLimit, serverPort } = startArgs;
  let configFile = startArgs.configFile;
  const localPort = startArgs.localPort ?? config.defaultLocalPort;
  const hfHome = `${config.projectDir}/hf`;

  const remoteWorkDir = `$HOME/${jobName}`;
  const remoteWorkDirScp = `~/${jobName}`;
  const remoteJobDetails = `${remoteWorkDir}/job_details.json`;

  // Set up dry-run temp dir and RemoteOps (needed before version discovery)
  const dryRunDir = startArgs.dryRun ? mkdtempSync(join(tmpdir(), "ivllm-dryrun-")) : undefined;
  const ops = makeRemoteOps(config, startArgs.dryRun, dryRunDir);

  // ── Pre-flight ──────────────────────────────────────────────────────────────
  // Check SSH connectivity early, before any remote operations
  if (!startArgs.dryRun) {
    console.log("Checking SSH connectivity...");
    const { exitCode: sshCheck } = await ops.runRemote("echo ok", { silent: true });
    if (sshCheck !== 0) {
      console.error("Error: Cannot connect to login node.");
      process.exit(1);
    }
    console.log("✓ SSH connectivity OK");
  }

  // Resolve config file: if --config provided, save to job store; if omitted, load from store.
  let usingStoredConfig = false;
  if (!startArgs.mock) {
    if (configFile) {
      try {
        saveJobConfig(jobName, configFile);
      } catch (e) {
        console.warn(`Warning: Could not save config to job store: ${(e as Error).message}`);
      }
    } else {
      const stored = jobConfigPath(jobName);
      if (!existsSync(stored)) {
        console.error(`Error: No --config provided and no stored config found for '${jobName}'.`);
        console.error(`  First run: ivllm start ${jobName} --config <path>`);
        process.exit(1);
      }
      configFile = stored;
      usingStoredConfig = true;
    }
  }

  // Resolve model, gpuCount, and vllmVersion:
  // - real mode: read from the vLLM config YAML and discover installed versions on remote
  // - mock mode: model comes from --model CLI flag; no remote version check
  let model: string;
  let gpuCount: number;
  let nodeCount: number;
  let maxModelLen: number | undefined;
  let enableAutoToolChoice: boolean | undefined;
  let enableReasoning: boolean | undefined;
  let vllmVersion: string;
  if (startArgs.mock) {
    model = startArgs.model!;
    gpuCount = startArgs.gpuCount ?? 1;
    nodeCount = 1;
    vllmVersion = "mock";
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
    maxModelLen = yamlConfig.maxModelLen;
    enableAutoToolChoice = yamlConfig.enableAutoToolChoice;
    enableReasoning = yamlConfig.enableReasoning;

    // Resolve which installed vLLM version to use based on min-vllm-version from the model config.
    // Scan the remote for installed versioned venvs and pick the highest that satisfies the minimum.
    const minVersion = yamlConfig.minVllmVersion;
    const installed = await listInstalledVersions(config, ops);
    if (installed.length === 0) {
      console.error(`Error: No vLLM installation found at ${config.projectDir}/ivllm/.`);
      console.error(minVersion
        ? `  Run: ivllm setup ${minVersion}`
        : `  Run: ivllm setup <version>`);
      process.exit(1);
    }
    if (minVersion) {
      const best = selectBestVersion(installed, minVersion);
      if (!best) {
        console.error(`Error: config requires vLLM >= ${minVersion} but installed versions are: ${installed.join(", ")}`);
        console.error(`  Run: ivllm setup ${minVersion}`);
        process.exit(1);
      }
      vllmVersion = best;
    } else {
      vllmVersion = semverSort(installed)[0]!;
    }
  }

  // Session state for shutdown sequence
  let shuttingDown = false;
  let slurmJobId: string | null = null;
  let tunnel: ChildProcess | null = null;
  let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  let crashDiagnosticsPrinted = false;

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

  async function printCrashDiagnostics(): Promise<void> {
    if (crashDiagnosticsPrinted || !slurmJobId) return;
    crashDiagnosticsPrinted = true;

    console.error("\n--- SLURM accounting ---");
    let stdout = "";
    for (let attempt = 0; attempt < 5; attempt++) {
      const result = await ops.runRemote(
        `${buildSacctDiagnosticsCommand(slurmJobId)} 2>/dev/null || true`,
        { silent: true }
      );
      stdout = result.stdout;
      if (stdout.trim() && sacctDiagnosticsSettled(stdout, slurmJobId)) break;
      if (attempt < 4) await sleep(2000);
    }
    if (stdout.trim()) {
      console.error(stdout);
      const localAccountingTmp = join(tmpdir(), `ivllm-${jobName}-slurm-accounting.txt`);
      writeFileSync(localAccountingTmp, stdout.endsWith("\n") ? stdout : `${stdout}\n`, "utf-8");
      try {
        await ops.copyFile(localAccountingTmp, `${remoteWorkDirScp}/slurm-accounting.txt`);
      } catch (error) {
        console.error(`(failed to save SLURM accounting snapshot to ~/${jobName}/slurm-accounting.txt: ${(error as Error).message})`);
      } finally {
        if (existsSync(localAccountingTmp)) unlinkSync(localAccountingTmp);
      }
    } else {
      console.error("(no sacct output available)");
    }
    console.error("--- end accounting ---");
    console.error(`Remote work dir : ~/${jobName}`);
    console.error(`Remote script   : ~/${jobName}/${jobName}.slurm.sh`);
    console.error(`Remote log      : ~/${jobName}/${jobName}.slurm.log`);
    console.error(`Remote ray logs : ~/${jobName}/ray-logs/`);
    console.error(`Remote sacct    : ~/${jobName}/slurm-accounting.txt`);
    console.error(`Remote details  : ~/${jobName}/job_details.json\n`);
  }

  // Register signal handlers immediately
  process.on("SIGINT", () => shutdown("interrupted (Ctrl+C)"));
  process.on("SIGTERM", () => shutdown("terminated"));

  console.log(`=== ivllm start (v${ivllmVersion}) ===`);
  console.log(`Job        : ${jobName}`);
  console.log(`Model      : ${model}`);
  console.log(`Config     : ${configFile ? `${configFile}${usingStoredConfig ? " (stored)" : ""}` : "(N/A — mock mode)"}`);
  console.log(`GPUs       : ${gpuCount}${nodeCount > 1 ? ` (${nodeCount} nodes × ${gpuCount / nodeCount} GPUs each)` : ""}`);
  if (nodeCount > 1) {
    console.log(`⚠ Multi-node job: ${nodeCount} nodes requested`);
  }
  console.log(`Local port : ${localPort}  |  Server port: ${serverPort}`);
  console.log("");

  // ── Venv check (after SSH is confirmed working) ─────────────────────────────
  if (!startArgs.dryRun) {
    console.log("Checking venv...");
    const venvDir = `${config.projectDir}/ivllm/${vllmVersion}`;
    const { exitCode: venvCheck } = await ops.runRemote(
      `test -f ${venvDir}/bin/activate`, { silent: true }
    );
    if (venvCheck !== 0) {
      console.error(`Error: vLLM venv not found at ${venvDir}. Run 'ivllm setup' first.`);
      process.exit(1);
    }
    console.log("✓ Venv check passed");
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
      const hfToken = config.hfToken ?? process.env["HF_TOKEN"] ?? "";
      const downloadCmd = `source ${config.projectDir}/ivllm/${vllmVersion}/bin/activate && HF_HOME=${hfHome}${hfToken ? ` HF_TOKEN=${hfToken}` : ""} hf download ${model}`;
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
        jobName, model, vllmVersion, hfHome,
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
  let lastSlurmPollTime = 0; // timestamp of last squeue/sacct call

  while (!shuttingDown) {
    await sleep(POLL_INTERVAL_MS);
    const { stdout } = await ops.runRemote(`cat ${remoteJobDetails} 2>/dev/null`, { silent: true });
    const details = parseJobDetails(stdout);

    if (!details) {
      // File missing — SLURM job may have died without updating it
      // Gate sacct call to Slurm poll interval
      if (Date.now() - lastSlurmPollTime >= SLURM_POLL_INTERVAL_MS) {
        lastSlurmPollTime = Date.now();
        const slurmState = await pollJobStatus(config, slurmJobId!);
        if (slurmState === "failed") {
          await printSlurmLog(config, remoteWorkDir, jobName);
          await printCrashDiagnostics();
          await shutdown("SLURM job failed unexpectedly", 1);
          return;
        }
      }
      continue;
    }

    // Issue #2a: while our lockfile shows "pending", report actual SLURM queue state
    // Gate squeue calls to at most once per SLURM_POLL_INTERVAL_MS (Isambard policy)
    if (details.status === "pending") {
      if (Date.now() - lastSlurmPollTime >= SLURM_POLL_INTERVAL_MS) {
        lastSlurmPollTime = Date.now();
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
      await printCrashDiagnostics();
      await shutdown(`vLLM ${details.status}`, 1);
      return;
    }

    if (details.status === "running") {
      rl.close();
      await onRunning(details);
      return;
    }
  }

  interface AssistantMenuOptions {
    model: string;
    localPort: number;
    maxModelLen?: number;
    toolCall?: boolean;
    reasoning?: boolean;
    shutdown: (reason: string, code?: number) => void;
  }

  interface MenuOption<T extends string> {
    key: T;
    label: string;
    input: string;
  }

  async function promptInput(question: string): Promise<string> {
    const rl = createInterface({ input: process.stdin, output: process.stdout, terminal: true });
    const answer = await new Promise<string>((resolve) => {
      rl.question(question, (value) => {
        resolve(value.trim());
        rl.close();
      });
    });
    return answer;
  }

  async function promptMenu<const T extends string>(title: string, options: readonly MenuOption<T>[]): Promise<T> {
    while (true) {
      console.log(title);
      for (const option of options) {
        console.log(`  [${option.input}] ${option.label}`);
      }
      console.log("");

      const answer = await promptInput("Selection: ");
      const selected = options.find((option) => option.input === answer);
      if (selected) return selected.key;

      console.log(`Invalid selection: ${answer || "(empty)"}. Please try again.\n`);
    }
  }

  async function launchAssistantMenu(opts: AssistantMenuOptions): Promise<void> {
    const { model, localPort, maxModelLen, toolCall, reasoning, shutdown } = opts;
    let cwd = process.cwd();
    const availableAssistants = getAvailableAssistants();
    const hasScoder = getScoderAvailable();
    const hasSbx = getSbxAvailable();

    if (availableAssistants.length === 0 && !hasSbx) {
      console.log("\nNo supported assistant launchers detected on PATH.");
      console.log("Showing opencode.ai config snippet instead:");
      console.log(formatOpencodeSnippet({
        model,
        localPort,
        maxModelLen,
        toolCall,
        reasoning,
      }));
      return;
    }

    while (true) {
      const targetChoice = await promptMenu(
        `\n🤖 AI coding assistant launcher\n📍 Working directory: ${cwd}\n\nChoose an assistant target:\n`,
        [
          { key: "opencode", label: "OpenCode", input: "1" },
          { key: "copilot", label: "GitHub Copilot", input: "2" },
          { key: "claude", label: "Claude Code", input: "3" },
          { key: "change-dir", label: "Change directory", input: "d" },
          { key: "show-snippet", label: "Show OpenCode config snippet", input: "s" },
          { key: "shutdown", label: "Shutdown ivllm", input: "0" },
        ] as const
      );

      if (targetChoice === "change-dir") {
        const newDir = await promptInput("\nEnter directory path (or press Enter to keep current): ");
        if (!newDir) continue;

        const nextCwd = resolve(newDir);
        if (!existsSync(nextCwd)) {
          console.log(`⚠️  Directory not found: ${newDir}. Keeping current directory.\n`);
          continue;
        }
        cwd = nextCwd;
        console.log(`✅ Changed directory to: ${cwd}\n`);
        continue;
      }

      if (targetChoice === "show-snippet") {
        console.log("\n📋 opencode.ai config snippet:");
        console.log(formatOpencodeSnippet({ model, localPort, maxModelLen, toolCall, reasoning }));
        console.log("");
        continue;
      }

      if (targetChoice === "shutdown") {
        await shutdown("user requested exit");
        return;
      }

      const assistant = targetChoice;
      const wrappers = getAvailableWrappers(assistant, availableAssistants, hasScoder, hasSbx);

      if (wrappers.length === 0) {
        console.log(`\n⚠️  No launch wrappers are available for ${getAssistantLabel(assistant)}.`);
        console.log("Install the local assistant binary for direct/scoder launch, or install sbx for sandbox launch.\n");
        continue;
      }

      const wrapperChoice = await promptMenu(
        `\n🎯 Target: ${getAssistantLabel(assistant)}\n📍 Working directory: ${cwd}\n\nChoose a wrapper:\n`,
        [
          ...wrappers.map((wrapper, index) => ({
            key: wrapper,
            label: wrapper === "none" ? "Direct launch" : wrapper.toUpperCase(),
            input: String(index + 1),
          })),
          { key: "back", label: "Back", input: "0" },
        ] as const
      );

      if (wrapperChoice === "back") continue;

      const wrapper = wrapperChoice;
      const action = await promptMenu(
        `\n🚀 ${getAssistantLabel(assistant)} via ${wrapper === "none" ? "direct launch" : wrapper.toUpperCase()}\n\nChoose an action:\n`,
        [
          { key: "launch", label: "Launch now", input: "1" },
          { key: "show", label: "Show copy-paste command", input: "2" },
          { key: "back", label: "Back", input: "0" },
        ] as const
      );

      if (action === "back") continue;

      let sandboxName = wrapper === "sbx" ? buildSandboxName(assistant, cwd) : undefined;
      let showCommands: string[];

      if (wrapper === "sbx") {
        try {
          const existing = findMatchingSandbox(listSbxSandboxes(), assistant, cwd);
          sandboxName = existing?.name ?? sandboxName;
          const launchCommand = buildLaunchCommand({
            assistant,
            wrapper,
            cwd,
            model,
            localPort,
            maxModelLen,
            toolCall,
            reasoning,
            sandboxName,
          });
          showCommands = existing
            ? [launchCommand]
            : [buildSandboxCreateCommand(assistant, cwd, sandboxName), launchCommand];
        } catch (error) {
          console.log(`\n⚠️  Failed to inspect sbx sandboxes: ${(error as Error).message}\n`);
          continue;
        }
      } else {
        showCommands = [
          buildLaunchCommand({
            assistant,
            wrapper,
            cwd,
            model,
            localPort,
            maxModelLen,
            toolCall,
            reasoning,
          }),
        ];
      }

      console.log("\n📋 Command:");
      for (const command of showCommands) {
        console.log(command);
      }
      console.log("");

      if (action === "show") continue;

      try {
        if (wrapper === "sbx") {
          const ensured = ensureSbxSandbox(assistant, cwd);
          sandboxName = ensured.sandboxName;
          if (ensured.created) {
            console.log(`✅ Created sandbox: ${sandboxName}`);
          }
        }

        const launchCommand = buildLaunchCommand({
          assistant,
          wrapper,
          cwd,
          model,
          localPort,
          maxModelLen,
          toolCall,
          reasoning,
          sandboxName,
        });

        const tmuxResult = spawnSync("tmux", ["new-window", "-n", assistant, "bash", "-lc", launchCommand], {
          stdio: "inherit",
        });
        if (tmuxResult.status !== 0) {
          console.log(`⚠️  Failed to launch ${getAssistantLabel(assistant)} in tmux. Run the command above manually.`);
          continue;
        }
      } catch (error) {
        console.log(`⚠️  Failed to launch ${getAssistantLabel(assistant)}. Error: ${(error as Error).message}`);
        continue;
      }

      console.log(`\n✅ ${getAssistantLabel(assistant)} launched. Return to menu when done.\n`);
      await promptInput("Press Enter to return to menu...");
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

    if (startArgs.noLaunch) {
      console.log("\n📋 opencode.ai config snippet:");
      console.log(formatOpencodeSnippet({
        model: details.model ?? model,
        localPort,
        maxModelLen,
        toolCall: enableAutoToolChoice,
        reasoning: enableReasoning,
      }));
      console.log("\nType 'exit' + Enter to stop, or press Ctrl+C\n");
    } else {
      await launchAssistantMenu({
        model: details.model ?? model,
        localPort,
        maxModelLen,
        toolCall: enableAutoToolChoice,
        reasoning: enableReasoning,
        shutdown,
      });
    }

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
            await printCrashDiagnostics();
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

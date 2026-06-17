import type {VllmConfig, SessionState, ResolvedSessionConfig, RemoteOps, EnvVarEntry, Config, MonitorRuntimeOpts} from './types.ts'
import {
  hfCachePath,
  parseJobDetails,
} from './job.ts';
import type { StartArgs, JobDetails } from "./types";
import { semverGte, semverSort } from './semver.ts';

import {
  parseVllmConfig,
  resolveGpuCount,
  parseEnvVars,
  jobConfigPath,
  writeStrippedConfig,
} from './vllm-config.ts';
import {
  sacctDiagnosticsSettled,
  buildSacctDiagnosticsCommand,
  submitJob,
  runInteractive,
  pollJobStatus,
  getSlurmQueueState,
} from './slurm.ts';
import { renderInferenceScript } from './templates/inference.ts';
import { renderMockInferenceScript } from './templates/mock-inference.ts';
import { spawnTunnel } from './ssh.ts';
import { launchAssistant } from './commands/agent.ts';
import { createInterface } from 'readline';
import { writeFileSync, existsSync, unlinkSync, mkdtempSync } from 'fs';
import { tmpdir } from 'os';
import { join, basename } from 'path';
// import { createHash } from 'crypto';

// function getHash(obj: Record<string, unknown>): string {
//   // Sort keys so identical configs produce identical hashes regardless of input order
//   const str = JSON.stringify(obj, Object.keys(obj).sort());
//   return createHash('sha256').update(str).digest('hex').slice(0, 16);
// }

// ── Constants ──────────────────────────────────────────────────────────────
const HEARTBEAT_INTERVAL_MS = 15_000;
const POLL_INTERVAL_MS = 5_000;
const SLURM_POLL_INTERVAL_MS = 60_000;

// ── Types ─────────────────────────────────────────────────────────────────


// ── Configuration resolution ──────────────────────────────────────────────

/**
 *
 * @param ops
 * @param config
 * @param startArgs
 */
export async function resolveSessionConfig(
  ops: RemoteOps,
  config: Config,
  startArgs: StartArgs,
): Promise<ResolvedSessionConfig> {
  let configFile = startArgs.configFile;
  let preCache = startArgs.preCache;
  let usingStoredConfig = false;
  if (!startArgs.mock) {
    if (!configFile) {
      const stored = jobConfigPath(startArgs.jobName);
      if (!existsSync(stored)) {
        throw new Error(
          `No --config provided and no stored config found for '${startArgs.jobName}'.\n  First run: ivllm start ${startArgs.jobName} --config <path>`,
        );
      }
      configFile = stored;
      usingStoredConfig = true;
    }
  }

  if (startArgs.mock) {
    return {
      configFile: configFile ?? '',
      usingStoredConfig: false,
      yamlConfig: {
        env: [],
        raw: {}
      },
      model: startArgs.model!,
      gpuCount: startArgs.gpuCount ?? 1,
      nodeCount: 1,
      maxModelLen: undefined,
      tensorParallelSize: 1,
      pipelineParallelSize: 1,
      enableAutoToolChoice: true,
      enableReasoning: true,
      vllmVersion: 'mock',
      envVars: [],
      preCache: preCache,
    };
  }

  const yamlConfig = parseVllmConfig(configFile!);
  if (!yamlConfig.model) {
    throw new Error(
      `'model' is required in the vLLM config file '${configFile}'.`,
    );
  }

  const model = yamlConfig.model;
  const { gpuCount, nodeCount } = resolveGpuCount(
    startArgs.gpuCount,
    yamlConfig,
  );
  const envVars = parseEnvVars(configFile!);

  const installed = await listInstalledVersions(config, ops);
  if (installed.length === 0) {
    throw new Error(
      `No vLLM installation found at ${config.projectDir}/ivllm/. Run 'ivllm setup <version>'.`,
    );
  }

  const minVersion = yamlConfig.minVllmVersion;
  const bestVersion = minVersion
    ? selectBestVersion(installed, minVersion)
    : semverSort(installed)[0];

  if (!bestVersion) {
    throw new Error(
      minVersion
        ? `Config requires vLLM >= ${minVersion} but installed versions are: ${installed.join(', ')}`
        : `No suitable vLLM version found.`,
    );
  }

  return {
    configFile: configFile!,
    usingStoredConfig,
    yamlConfig,
    model,
    gpuCount,
    nodeCount,
    maxModelLen: yamlConfig.maxModelLen,
    tensorParallelSize: yamlConfig.tensorParallelSize ?? 1,
    pipelineParallelSize: yamlConfig.pipelineParallelSize ?? 1,
    enableAutoToolChoice: yamlConfig.enableAutoToolChoice ?? true,
    enableReasoning: yamlConfig.enableReasoning ?? true,
    vllmVersion: bestVersion,
    envVars,
    preCache,
  };
}

// ── Shared utilities ──────────────────────────────────────────────────────

/**
 *
 * @param startArgs
 * @param ops
 * @param config
 */
export async function preFlight(
  startArgs: StartArgs,
  ops: RemoteOps,
  config: Config,
): Promise<void> {
  const localPort = startArgs.localPort ?? config.defaultLocalPort;
  if (!startArgs.dryRun) {
    console.log('Checking SSH connectivity...');
    const { exitCode: sshCheck } = await ops.runRemote('echo ok', {
      env:[], silent: true,
    });
    if (sshCheck !== 0) {
      console.error('Error: Cannot connect to login node.');
      process.exit(1);
    }
    console.log('✓ SSH connectivity OK');
    console.log(`Checking local port ${localPort}...`);
    const portInUse = await isLocalPortInUse(localPort);
    if (portInUse) {
      console.error(
        `Error: Local port ${localPort} is already in use by ${portInUse.process} (pid ${portInUse.pid}).`,
      );
      process.exit(1);
    }
    console.log(`✓ Port ${localPort} is free`);
  }
}

/**
 *
 * @param ops
 * @param config
 * @param startArgs
 * @param hfHome
 * @param model
 * @param vllmVersion
 */
export async function ensureModelDownloaded(
  ops: RemoteOps,
  config: Config,
  startArgs: StartArgs,
  hfHome: string,
  model: string,
  vllmVersion: string,
): Promise<void> {
  const cachePath = hfCachePath(hfHome, model);
  if (startArgs.dryRun) {
    console.log(`[dry-run] HF cache check skipped (would check: ${cachePath})`);
  } else if (startArgs.mock) {
    console.log(`[mock] Model download skipped`);
  } else {
    const { exitCode: cacheCheck } = await ops.runRemote(
      `test -d ${cachePath}`,
      { env:[], silent: true },
    );
    if (cacheCheck === 0) {
      console.log(`✓ Model cached at ${cachePath}`);
    } else {
      console.log(`Downloading model ${model} to ${hfHome} on login node...`);
      await ops.runRemote(
        `mkdir -p ${hfHome} && chmod g+w ${hfHome} 2>/dev/null || true`,
        { env:[], silent: true },
      );
      const hfToken = config.hfToken ?? process.env['HF_TOKEN'] ?? '';
      const downloadCmd = `umask 0002 && source ${config.projectDir}/ivllm/${vllmVersion}/bin/activate && HF_HOME=${hfHome}${hfToken ? ` HF_TOKEN=${hfToken}` : ''} hf download ${model}`;
      const { exitCode: dlCode } = await ops.runRemote(downloadCmd);
      if (dlCode !== 0) {
        console.error('Error: Model download failed.');
        process.exit(1);
      }
      console.log('✓ Model downloaded');
    }
  }
}

/**
 *
 * @param ops
 * @param jobName
 * @param remoteWorkDir
 * @param remoteJobDetails
 * @param dryRun
 */
export async function createJobLockfile(
  ops: RemoteOps,
  jobName: string,
  remoteWorkDir: string,
  remoteJobDetails: string,
  dryRun: boolean,
): Promise<void> {
  if (dryRun) {
    console.log(
      `[dry-run] Lockfile creation skipped (would create: ${remoteJobDetails})`,
    );
  } else {
    console.log('Creating job working directory and lockfile...');
    await ops.runRemote(`mkdir -p ${remoteWorkDir}`, { env:[], silent: true });
    const pendingJson = JSON.stringify({
      status: 'pending',
      job_name: jobName,
    });
    const { exitCode: lockCode } = await ops.runRemote(
      `set -C; echo '${pendingJson}' > ${remoteJobDetails}`,
      { env:[], silent: true },
    );
    if (lockCode !== 0) {
      console.error(
        `Error: Job '${jobName}' already exists (lockfile present). Use 'ivllm stop ${jobName}' to clean up.`,
      );
      process.exit(1);
    }
    console.log('✓ Lockfile created');
  }
}

// ── Shutdown & diagnostics ────────────────────────────────────────────────

/**
 *
 * @param state
 */
export async function printSlurmLog(state: SessionState): Promise<void> {
  console.error('\n--- SLURM log ---');
  const { stdout } = await state.ops.runRemote(
    `tail -50 ${state.remoteWorkDir}/${state.jobName}.slurm.log 2>/dev/null`,
    { env:[], silent: true },
  );
  if (stdout) console.error(stdout);
  console.error('--- end log ---\n');
}

/**
 *
 * @param state
 */
export async function printCrashDiagnostics(
  state: SessionState,
): Promise<void> {
  if (state.crashDiagnosticsPrinted || !state.slurmJobId) return;
  state.crashDiagnosticsPrinted = true;

  console.error('\n--- SLURM accounting ---');
  let stdout = '';
  for (let attempt = 0; attempt < 5; attempt++) {
    const result = await state.ops.runRemote(
      `${buildSacctDiagnosticsCommand(state.slurmJobId)} 2>/dev/null || true`,
      { env:[],  silent: true },
    );
    stdout = result.stdout;
    if (stdout.trim() && sacctDiagnosticsSettled(stdout, state.slurmJobId))
      break;
    if (attempt < 4) await sleep(2000);
  }
  if (stdout.trim()) {
    console.error(stdout);
    const localAccountingTmp = join(
      tmpdir(),
      `ivllm-${state.jobName}-slurm-accounting.txt`,
    );
    writeFileSync(
      localAccountingTmp,
      stdout.endsWith('\n') ? stdout : `${stdout}\n`,
      'utf-8',
    );
    try {
      await state.ops.copyFile(
        localAccountingTmp,
        `${state.remoteWorkDir}/${state.jobName}.slurm-accounting.txt`,
      );
    } catch (error) {
      console.error(
        `(failed to save SLURM accounting snapshot: ${(error as Error).message})`,
      );
    } finally {
      if (existsSync(localAccountingTmp)) unlinkSync(localAccountingTmp);
    }
  } else {
    console.error('(no sacct output available)');
  }
  console.error('--- end accounting ---\n');
  console.error(`Remote work dir : ~/${state.jobName}`);
  console.error(
    `Remote script   : ~/${state.jobName}/${state.jobName}.slurm.sh`,
  );
  console.error(
    `Remote log      : ~/${state.jobName}/${state.jobName}.slurm.log`,
  );
  console.error(`Remote ray logs : ~/${state.jobName}/ray-logs/`);
  console.error(`Remote sacct    : ~/${state.jobName}/slurm-accounting.txt`);
  console.error(`Remote details  : ~/${state.jobName}/job_details.json\n`);
}

/**
 *
 * @param state
 * @param reason
 * @param exitCode
 */
export async function shutdown(
  state: SessionState,
  reason: string,
  exitCode = 0,
): Promise<void> {
  if (state.shuttingDown) return;
  state.shuttingDown = true;
  console.log(`\n⏹  Shutting down: ${reason}`);

  if (state.heartbeatTimer) clearInterval(state.heartbeatTimer);

  if (state.slurmJobId) {
    process.stdout.write('  Cancelling SLURM job...');
    await state.ops
      .runRemote(`scancel ${state.slurmJobId}`, { env:[], silent: true })
      .catch(() => {});
    console.log(' done');
  }

  if (state.tunnel) {
    process.stdout.write('  Closing SSH tunnel...');
    state.tunnel.kill();
    console.log(' done');
  }

  process.stdout.write('  Removing lockfile...');
  await state.ops
    .runRemote(`rm -f ${state.remoteJobDetails}`, { env:[], silent: true })
    .catch(() => {});
  console.log(' done');

  console.log('✓ Session ended');
  process.exit(exitCode);
}

// ── Version helpers ───────────────────────────────────────────────────────

/**
 *
 * @param installed
 * @param minVersion
 */
export function selectBestVersion(
  installed: string[],
  minVersion: string,
): string | null {
  const candidates = installed.filter((v) => semverGte(v, minVersion));
  if (candidates.length === 0) return null;
  return semverSort(candidates)[0]!;
}

/**
 *
 * @param config
 * @param ops
 */
export async function listInstalledVersions(
  config: Config,
  ops: RemoteOps,
): Promise<string[]> {
  const { stdout } = await ops.runRemote(
    `ls -d ${config.projectDir}/ivllm/*/bin 2>/dev/null | sed 's|.*/ivllm/||; s|/bin||'`,
    { env: [], silent: true },
  );
  return stdout
    .trim()
    .split('\n')
    .filter((v) => v && /^\d+\.\d+/.test(v));
}

// ── Utility functions ─────────────────────────────────────────────────────

/**
 *
 */
export function timestamp(): string {
  return new Date().toTimeString().slice(0, 8);
}

/**
 *
 * @param ms
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 *
 * @param port
 */
export async function isLocalPortInUse(
  port: number,
): Promise<{ pid: string; process: string } | null> {
  const { execFile } = await import('child_process');
  return new Promise((resolve) => {
    execFile('lsof', ['-ti', `:${port}`, '-sTCP:LISTEN'], (err, stdout) => {
      if (err || !stdout.trim()) {
        resolve(null);
        return;
      }
      const pid = stdout.trim().split('\n')[0] as string;
      execFile('ps', ['-p', pid, '-o', 'comm='], (_err2, psOut) => {
        const process = psOut?.trim() || 'unknown';
        resolve({ pid, process });
      });
    });
  });
}

// ── Unified session pipeline ──────────────────────────────────────────────

/**
 *
 * @param config
 * @param startArgs
 * @param isInteractive
 * @param ops
 */
export async function runInferenceSession(
  config: Config,
  startArgs: StartArgs,
  isInteractive: boolean,
  ops: RemoteOps,
): Promise<void> {
  const { version: ivllmVersion } = await import('../package.json');
  const { jobName, timeLimit, serverPort } = startArgs;
  const localPort = startArgs.localPort ?? config.defaultLocalPort;
  const hfHome = `${config.projectDir}/hf`;
  const remoteWorkDir = `$HOME/${jobName}`;
  const remoteWorkDirScp = `~/${jobName}`;
  const remoteJobDetails = `${remoteWorkDir}/job_details.json`;

  // ── 1. Pre-flight ──────────────────────────────────────────────────────
  await preFlight(startArgs, ops, config);

  // ── 2. Resolve config ─────────────────────────────────────────────────
  const resolved = await resolveSessionConfig(ops, config, startArgs);
  const {
    model,
    gpuCount,
    nodeCount,
    vllmVersion,
    envVars,
    configFile,
    maxModelLen,
    tensorParallelSize,
    pipelineParallelSize,
    enableAutoToolChoice,
    enableReasoning,
    usingStoredConfig,
    preCache,
  } = resolved;

  //const cacheKey = getHash({ model, gpuCount, nodeCount, maxModelLen });
  // cache key is a filepath. Will have .tar.gz added.
  const cacheKey = `${model}/${tensorParallelSize}/${pipelineParallelSize}/${maxModelLen ?? 'undefined'}/cached`;

  // ── 3. Build session state ────────────────────────────────────────────
  const sessionState: SessionState = {
    jobName,
    config,
    ops,
    remoteWorkDir,
    remoteJobDetails,
    slurmJobId: null,
    tunnel: null,
    heartbeatTimer: null,
    crashDiagnosticsPrinted: false,
    shuttingDown: false,
  };

  // ── 4. Signal handlers ────────────────────────────────────────────────
  process.on('SIGINT', () => shutdown(sessionState, 'interrupted (Ctrl+C)'));
  process.on('SIGTERM', () => shutdown(sessionState, 'terminated'));

  // ── 5. Startup info ───────────────────────────────────────────────────
  console.log(
    `=== ivllm ${isInteractive ? 'interactive' : 'start'} (v${ivllmVersion}) ===`,
  );
  console.log(`Job        : ${jobName}`);
  console.log(`Model      : ${model}`);
  console.log(`Cached as  : ${cacheKey}`);
  console.log(
    `Config     : ${configFile ? `${configFile}${usingStoredConfig ? ' (stored)' : ''}` : '(N/A — mock mode)'}`,
  );
  console.log(
    `GPUs       : ${gpuCount}${nodeCount > 1 ? ` (${nodeCount} nodes × ${gpuCount / nodeCount} GPUs each)` : ''}`,
  );
  if (nodeCount > 1) {
    console.log(`⚠ Multi-node job: ${nodeCount} nodes requested`);
  }
  console.log(`Local port : ${localPort}  |  Server port: ${serverPort}`);
  console.log('');

  // ── 6. Venv check ─────────────────────────────────────────────────────
  if (!startArgs.dryRun) {
    console.log('Checking venv...');
    const venvDir = `${config.projectDir}/ivllm/${vllmVersion}`;
    const { exitCode: venvCheck } = await ops.runRemote(
      `test -f ${venvDir}/bin/activate`,
      { silent: true },
    );
    if (venvCheck !== 0) {
      console.error(
        `Error: vLLM venv not found at ${venvDir}. Run 'ivllm setup' first.`,
      );
      await shutdown(sessionState, 'venv not found', 1);
    }
    console.log('✓ Venv check passed');
  }

  // ── 7. Model download ─────────────────────────────────────────────────
  await ensureModelDownloaded(
    ops,
    config,
    startArgs,
    hfHome,
    model,
    vllmVersion,
  );

  // ── 8. Lockfile ───────────────────────────────────────────────────────
  await createJobLockfile(
    ops,
    jobName,
    remoteWorkDir,
    remoteJobDetails,
    startArgs.dryRun,
  );

  // ── 9. Generate script ────────────────────────────────────────────────
  const remoteConfigFile = configFile
    ? `${remoteWorkDir}/${basename(configFile)}`
    : undefined;
  const remoteScriptPath = `${remoteWorkDir}/${jobName}.slurm.sh`;
  const remoteConfigFileScp = configFile
    ? `${remoteWorkDirScp}/${basename(configFile)}`
    : undefined;
  const remoteScriptPathScp = `${remoteWorkDirScp}/${jobName}.slurm.sh`;

  const script = startArgs.mock
    ? renderMockInferenceScript({
        jobName,
        model,
        workDir: remoteWorkDir,
        serverPort,
        timeLimit,
      })
    : renderInferenceScript({
        jobName,
        model,
        vllmVersion,
        hfHome,
        configFileName: configFile ? basename(configFile) : '',
        workDir: remoteWorkDir,
        serverPort,
        gpuCount,
        nodeCount,
        timeLimit,
        envVars,
        isInteractive,
        cacheKey,
        preCache,
      });

  const localScriptTmp = join(tmpdir(), `ivllm-${jobName}.slurm.sh`);
  writeFileSync(localScriptTmp, script, 'utf-8');

  // ── 10. Copy & submit ─────────────────────────────────────────────────
  const dryRunDir = startArgs.dryRun
    ? mkdtempSync(join(tmpdir(), 'ivllm-dryrun-'))
    : undefined;

  try {
    console.log('Copying files to login node...');
    if (remoteConfigFileScp && configFile) {
      const strippedConfigTmp = writeStrippedConfig(configFile);
      try {
        await ops.copyFile(strippedConfigTmp, remoteConfigFileScp);
      } finally {
        unlinkSync(strippedConfigTmp);
      }
    }
    await ops.copyFile(localScriptTmp, remoteScriptPathScp);
    console.log('✓ Files copied');

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

    const monitorOpts: MonitorRuntimeOpts = {
      localPort,
      serverPort,
      config,
      model,
      maxModelLen,
      enableAutoToolChoice,
      enableReasoning,
      isInteractive: isInteractive || !!startArgs.mock,
    };

    if (isInteractive || startArgs.mock) {
      console.log('Running interactive job...');
      await runInteractive(
        config,
        {
          jobName,
          model,
          vllmVersion,
          hfHome,
          configFileName: configFile ? basename(configFile) : '',
          workDir: remoteWorkDir,
          serverPort,
          gpuCount,
          nodeCount,
          timeLimit,
          isInteractive,
          envVars,
          cacheKey,
          preCache,
        },
        remoteScriptPath,
        sessionState,
        startArgs,
        monitorOpts,
        monitorSession,
      );
    } else {
      console.log('Submitting SLURM job...');
      await submitJob(
        config,
        remoteScriptPath,
        sessionState,
        startArgs,
        monitorOpts,
        preCache ? detachSession : monitorSession,
      );
    }
  } finally {
    if (existsSync(localScriptTmp)) unlinkSync(localScriptTmp);
  }
}

// ── Monitor helpers ───────────────────────────────────────────────────────



async function detachSession(
  sessionState: SessionState,
  startArgs: StartArgs,
  opts: MonitorRuntimeOpts,
): Promise<void> {
  console.log(
    `\nBatch job submintted: to cancel: ivllm stop ${startArgs.jobName}...\n`,
  );
  return;
}

/**
 *
 * @param sessionState
 * @param startArgs
 * @param opts
 */
async function monitorSession(
  sessionState: SessionState,
  startArgs: StartArgs,
  opts: MonitorRuntimeOpts,
): Promise<void> {
  const {
    localPort,
    serverPort,
    config,
    model,
    maxModelLen,
    enableAutoToolChoice,
    enableReasoning,
  } = opts;
  const { jobName } = sessionState;

  console.log("\nMonitoring job status (Ctrl+C or type 'exit' to stop)...\n");

  const rl = createInterface({ input: process.stdin, terminal: false });
  rl.on('line', (line) => {
    if (line.trim().toLowerCase() === 'exit') {
      rl.close();
      shutdown(sessionState, 'user requested exit');
    }
  });

  let lastStatus = 'pending';
  let lastSlurmQueueState = '';
  let logLineOffset = 0;
  let lastSlurmPollTime = 0;

  while (!sessionState.shuttingDown) {
    await sleep(POLL_INTERVAL_MS);
    const { stdout } = await sessionState.ops.runRemote(
      `cat ${sessionState.remoteJobDetails} 2>/dev/null`,
      { silent: true },
    );
    const details = parseJobDetails(stdout);

    if (!details) {
      if (Date.now() - lastSlurmPollTime >= SLURM_POLL_INTERVAL_MS) {
        lastSlurmPollTime = Date.now();
        const slurmState = await pollJobStatus(
          config,
          sessionState.slurmJobId!,
        );
        if (slurmState === 'failed') {
          await printSlurmLog(sessionState);
          await printCrashDiagnostics(sessionState);
          await shutdown(sessionState, 'SLURM job failed unexpectedly', 1);
          return;
        }
      }
      continue;
    }

    if (details.status === 'pending') {
      if (Date.now() - lastSlurmPollTime >= SLURM_POLL_INTERVAL_MS) {
        lastSlurmPollTime = Date.now();
        const queueState = await getSlurmQueueState(
          config,
          sessionState.slurmJobId!,
        );
        if (queueState) {
          const msg =
            queueState.state === 'PENDING'
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
      if (details.status === 'initialising') {
        console.log(
          `  [${timestamp()}] Job allocated — vLLM is starting up...`,
        );
      } else if (details.status !== 'pending') {
        console.log(`  [${timestamp()}] Status: ${details.status}`);
      }
      lastStatus = details.status;
    }

    if (!opts.isInteractive && details.status === 'initialising') {
      const slurmLogPath = `${sessionState.remoteWorkDir}/${jobName}.slurm.log`;
      const { stdout: newLines } = await sessionState.ops.runRemote(
        `tail -n +${logLineOffset + 1} ${slurmLogPath} 2>/dev/null`,
        { silent: true },
      );
      if (newLines.trim()) {
        const lines = newLines.split('\n').filter((l) => l.trim());
        for (const line of lines) {
          console.log(`  | ${line}`);
        }
        logLineOffset += lines.length;
      }
    }

    if (details.status === 'failed' || details.status === 'timeout') {
      if (details.error) console.error(`  Error: ${details.error}`);
      await printSlurmLog(sessionState);
      await printCrashDiagnostics(sessionState);
      await shutdown(sessionState, `vLLM ${details.status}`, 1);
      return;
    }

    if (details.status === 'running') {
      rl.close();
      await onRunning(details);
      return;
    }
  }

  /**
   *
   * @param details
   */
  async function onRunning(details: JobDetails): Promise<void> {
    const computeHost = details.compute_hostname!;
    console.log(`\n✓ vLLM is running on ${computeHost}:${serverPort}`);

    const tunnel = spawnTunnel(config, localPort, computeHost, serverPort);
    sessionState.tunnel = tunnel;
    tunnel.on('exit', (code) => {
      if (!sessionState.shuttingDown)
        shutdown(
          sessionState,
          `SSH tunnel exited unexpectedly (code ${code})`,
          1,
        );
    });

    await sleep(2000);

    console.log(`\n🚀 OpenAI API endpoint: http://localhost:${localPort}/v1`);
    console.log(`   Model: ${details.model ?? model}`);

    await launchAssistant({
      model: details.model ?? model,
      localPort,
      maxModelLen,
      toolCall: enableAutoToolChoice,
      reasoning: enableReasoning,
      shutdown: (reason: string, exitCode = 0) =>
        shutdown(sessionState, reason, exitCode),
    });

    const heartbeatTimer = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:${localPort}/health`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
      } catch (e) {
        if (!sessionState.shuttingDown) {
          console.error(`\nHeartbeat failed: ${(e as Error).message}`);
          await printSlurmLog(sessionState);
          await printCrashDiagnostics(sessionState);
          shutdown(sessionState, 'vLLM heartbeat failed', 1);
        }
      }
    }, HEARTBEAT_INTERVAL_MS);
    sessionState.heartbeatTimer = heartbeatTimer;
  }
}

import { writeFileSync, unlinkSync, existsSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { loadCredentials, assertConfigured } from '../config.ts';
import { renderSetupScript } from '../templates/setup.ts';
import {
  submitJob,
  pollJobStatus,
  getJobLog,
  getSlurmQueueState,
} from '../slurm.ts';
import { checkSSH, makeRemoteOps } from '../remote-ops.ts';
import { ProcessState } from '../types.ts';
import { detachSession } from '../monitors.ts';
import { makeSimplePaths } from '../job.ts';
import { makeLocalOps } from '../local-ops.ts';

const POLL_INTERVAL_MS = 60_000; // Isambard policy: do not poll Slurm scheduler more than once per minute

// TODO: --dry-run flag

/**
 *
 * @param args
 */
export async function cmdSetup(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: ivllm setup <version>

Options:
  <version>             vLLM version to install (e.g. 0.19.1)
  --help, -h            Show this help message

Examples:
  ivllm setup 0.19.1
`);
    return;
  }

  const vllmVersion = args[0];
  if (!vllmVersion || vllmVersion.startsWith('--')) {
    console.error('Error: vLLM version is required.');
    console.error('Usage: ivllm setup <version>  (e.g. ivllm setup 0.19.1)');
    process.exit(1);
  }

  const config = loadCredentials();
  try {
    assertConfigured(config);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  const ops = makeRemoteOps(config, false);
  const paths = makeSimplePaths(config, vllmVersion);
  const venvDir = paths.remoteProjectVllmVersionDir;
  const remoteSetupDir = `${paths.remoteHomeDir}/.config/ivllm/${vllmVersion}`;
  const remoteSetupScript = `${remoteSetupDir}/slurm.sh`;
  const remoteSetupLog = `${remoteSetupDir}/setup.log`;

  console.log(`=== ivllm setup (version ${__VERSION__}) ===`);
  console.log(`Login node  : ${config.loginHost}`);
  console.log(`vLLM        : ${vllmVersion}`);
  console.log(`Install dir : ${venvDir}`);
  console.log('');

  // ── 3. Build session state ────────────────────────────────────────────

  const sessionState = new ProcessState({
    sessionName: 'vllm install',
    vllmVersion,
    ops,
    paths,
  });

  // Pre-flight: check SSH connectivity
  ops.checkSSH();

  // Check if versioned venv already exists

  const { exitCode: venvCheck } = await ops.runRemote(`test -d ${venvDir}/bin`);
  if (venvCheck === 0) {
    console.log(`✓ vLLM ${vllmVersion} already installed at ${venvDir}`);
    console.log('  Delete the directory first to reinstall.');
    return;
  }

  // Render and copy setup script to LOGIN
  const script = renderSetupScript(sessionState, remoteSetupLog);

  const localTmp = join(tmpdir(), 'ivllm-setup.slurm.sh');
  writeFileSync(localTmp, script, 'utf-8');

  try {
    console.log('Copying setup script to login node...');
    await ops.runRemote(`mkdir -p ${remoteSetupDir})`);
    await ops.copyFile(localTmp, remoteSetupScript);
    console.log('✓ Script copied');

    // Submit SLURM job
    console.log('Submitting SLURM setup job...');

    //TODO: This is using old approach and could be refactored using a monitor.

    await submitJob(remoteSetupScript, sessionState);
    const jobId = sessionState.slurmJobId!;
    console.log(`✓ SLURM job submitted: ${jobId}`);

    // Truncate the log so we only stream this run's output
    await ops.runRemote(`truncate -s 0 ${remoteSetupLog} 2>/dev/null || true`);

    // Wait for the job to leave the queue (PENDING → RUNNING)
    console.log('Waiting for compute node allocation...');
    while (true) {
      const queueState = await getSlurmQueueState(ops, jobId);
      if (!queueState || queueState.state !== 'PENDING') break;
      await sleep(POLL_INTERVAL_MS);
    }

    console.log(
      '✓ Job started — streaming setup log (this may take 10–20 minutes)...',
    );
    console.log('─'.repeat(60));

    // Stream the remote log to stdout in real time
    // Give the job a moment to start writing the log before tailing
    await sleep(3_000);
    const tail = ops.tailRemoteLog(remoteSetupLog, '  ');

    // Poll for job completion while log streams in the background
    let state = await pollJobStatus(ops, jobId);
    while (state === 'running') {
      await sleep(POLL_INTERVAL_MS);
      state = await pollJobStatus(ops, jobId);
    }

    // Give tail a moment to flush remaining lines before stopping
    await sleep(2_000);
    tail.stop();
    console.log('─'.repeat(60));

    // Fetch log
    const log = await getJobLog(ops, remoteSetupLog);

    if (state === 'failed') {
      console.error('✗ Setup job failed. Log output:');
      console.error(log);
      process.exit(1);
    }

    if (!log.includes('IVLLM_SETUP_SUCCESS')) {
      console.error(
        '✗ Setup job completed but success marker not found. Log output:',
      );
      console.error(log);
      process.exit(1);
    }

    // Validate venv exists
    const { exitCode: finalCheck } = await ops.runRemote(
      `test -d ${venvDir}/bin`,
    );
    if (finalCheck !== 0) {
      console.error(`✗ venv not found at ${venvDir} after setup.`);
      process.exit(1);
    }

    console.log(`✓ vLLM ${vllmVersion} installation complete`);
    const versionLine = log.split('\n').find((l) => l.startsWith('vllm'));
    if (versionLine) console.log(`  ${versionLine}`);
  } finally {
    if (existsSync(localTmp)) unlinkSync(localTmp);
  }
}

/**
 *
 * @param ms
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

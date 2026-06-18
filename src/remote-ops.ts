import { copyFileSync, mkdtempSync } from 'fs';
import { basename, join } from 'path';
import EventEmitter from 'node:events';
import { spawn } from 'child_process';
import readline from 'readline';
import { semverGte, semverSort } from './semver.ts';

import type {
  CloseableEventEmitter,
  Credentials,
  RemoteOps,
  InferenceJobOptions,
  SessionState,
  RunRemoteOptions,
  RunRemoteResult,
} from './types.ts';

/**
 * Returns real RemoteOps that execute SSH/SCP, or dry-run ops that print
 * what would happen and copy files to a local preview directory.
 * @param config
 * @param dryRun
 * @param dryRunDir
 */
export function makeRemoteOps(config: Credentials, dryRun: boolean): RemoteOps {
  if (!dryRun) {
    return {
      runRemote: (cmd, opts) => runRemote(config, cmd, opts),
      copyFile: (local, remote) => copyFile(config, local, remote),
      streamSrun: (cmd, sessionState, opts) =>
        streamSrun(config, cmd, opts, sessionState),
      tailRemoteLog: (remote, prefix) => tailRemoteLog(config, remote, prefix),
      spawnTunnel: (localPort, remoteHost, remotePort) =>
        spawnTunnel(config, localPort, remoteHost, remotePort),
      matchVllmVersion: (minVllmVersion) =>
        matchVllmVersion(config, minVllmVersion),
    };
  }

  // Mock all network actions for E2E testing.
  return {
    async runRemote(command: string, opts) {
      console.log(`  [dry-run] Would run remotely:\n    ${command}`);
      return {
        exitCode: 0,
        stdout: command.startsWith('sbatch')
          ? 'Submitted batch job 123456'
          : command.startsWith('squeue')
            ? 'test-state test-reason'
            : command.startsWith('sacct')
              ? '123456 RUNNING'
              : command.startsWith('cat')
                ? 'lockfile' // needs json example
                : command.startsWith('ls -d')
                  ? '0.99.99' // version info
                  : '',
      };
    },
    async copyFile(localPath, remotePath) {
      const dryRunDir = mkdtempSync(join(process.cwd(), 'ivllm-dryrun-'));
      const destName = basename(remotePath);
      const dest = join(dryRunDir!, destName);
      copyFileSync(localPath, dest);
      console.log(`  [dry-run] Would scp: ${localPath} → ${remotePath}`);
      console.log(`           (preview: ${dest})`);
    },
    streamSrun(command, sessionState, opts) {
      console.log(`  [dry-run] Would stream remotely:\n    ${command}`);
      console.log(`srun: job 123456`);
      sessionState.slurmJobId = '123456';
      return createMockSSh('streaming srun', 2222);
    },
    tailRemoteLog(remote, prefix) {
      console.log(
        (prefix ?? '') + `  [dry-run] dummy log entry from ${remote}`,
      );
      return { stop: () => {} };
    },
    spawnTunnel(local, remoteHost, remotePort) {
      console.log(
        `  [dry-run] mock SSH tunnel created with ${local}:${remoteHost}:${remotePort}`,
      );

      // 1. Create the main process and its standard I/O stubs as plain EventEmitters
      return createMockSSh('tunnel', 1111);
    },
    async matchVllmVersion(minVllmVersion) {
      return minVllmVersion;
    },
  };
}

// ======================
// HELPERS
// ======================

function createMockSSh(name: string, pid: number): CloseableEventEmitter {
  const mockSshTunnel = Object.assign(new EventEmitter(), {
    stdin: Object.assign(new EventEmitter(), { write: () => true }),
    stdout: Object.assign(new EventEmitter(), {
      pipe: (destination: EventEmitter) => destination,
    }),
    stderr: Object.assign(new EventEmitter(), {
      pipe: (destination: EventEmitter) => destination,
    }),
    // 2. Add required identity properties so it looks like a running process
    pid: pid,
    exitCode: null, // Stays null so it looks permanently alive
    kill: () => {
      console.log(`  [dry-run] Shutting down ${name}`);
      mockSshTunnel.emit('close', 0);
      return true;
    },
  });
  return mockSshTunnel;
}

// Reusable SSH multiplexing options. The first connection spawns a
// background ControlMaster; subsequent connections within 10 minutes
// reuse the existing socket, avoiding repeated handshakes and login
// rate-limits on busy HPC login nodes.
const SSH_MUX_OPTS = [
  '-o',
  'ControlMaster=auto',
  '-o',
  'ControlPersist=600',
  '-o',
  'ControlPath=/tmp/ivllm-ssh-%r@%h:%p',
] as const;

/**
 * Run a command on the LOGIN node via SSH collecting stdout in a string.
 * Returns a promise that resolves with the exit code and the stdout.
 * @param config
 * @param command
 * @param options
 * @param options.env
 * @param options.silent
 */
function runRemote(
  config: Credentials,
  command: string,
  options: RunRemoteOptions = { env: [], silent: true },
): Promise<RunRemoteResult> {
  return new Promise((resolve, reject) => {
    const target = `${config.username}@${config.loginHost}`;
    const envPrefix =
      options.env.map((v) => `${v.key}=${v.value}`).join(' ') + ' ';
    const fullCommand = (envPrefix + command).trim();

    const proc = spawn(
      'ssh',
      [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, fullCommand],
      {
        stdio: options.silent
          ? ['ignore', 'pipe', 'pipe']
          : ['ignore', 'inherit', 'inherit'],
      },
    );

    let stdout = '';
    if (options.silent) {
      proc.stdout?.on('data', (chunk: Buffer) => {
        stdout += chunk.toString();
      });
    }

    proc.on('error', reject);
    proc.on('close', (code) =>
      resolve({ exitCode: code ?? 1, stdout: stdout.trim() }),
    );
  });
}

// Executes remote task and streams output to local terminal.
// observes the output looking for a job id and updates session state.
function streamSrun(
  config: Credentials,
  command: string,
  options: RunRemoteOptions = { env: [], silent: false },
  sessionState: SessionState,
): CloseableEventEmitter {
  const target = `${config.username}@${config.loginHost}`;
  const envPrefix =
    options.env.map((v) => `${v.key}=${v.value}`).join(' ') + ' ';
  const fullCommand = (envPrefix + command).trim();

  // Use -t for pseudo-tty streaming to bypass log buffering
  const proc = spawn(
    'ssh',
    ['-t', ...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, fullCommand],
    {
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );

  // Use readline interface to parse logs cleanly line by line
  const rl = readline.createInterface({
    input: proc.stdout!,
    terminal: false,
  });

  // let idReceived = false;
  rl.on('line', (line) => {
    // Print to local console unless silent
    if (!options.silent) {
      console.log(line);
    }
    // See if we can find the slurm job id in the output
    const srunMatch = line.match(/srun: (?:job|Job) (\d+)/i);
    if (srunMatch) {
      const foundId = srunMatch[1];
      // since this is passed by reference this should update everywhere
      sessionState.slurmJobId = foundId as string;
    }
  });

  proc.stderr?.on('data', (chunk: Buffer) => {
    if (!options.silent) {
      process.stderr.write(chunk.toString());
    }
  });

  return proc;
}

/**
 * Copy a local file to a path on the LOGIN node via scp.
 * Returns a promise that resolves when the copy is complete.
 * @param config
 * @param localPath
 * @param remotePath
 */
function copyFile(
  config: Credentials,
  localPath: string,
  remotePath: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const target = `${config.username}@${config.loginHost}:${remotePath}`;
    const proc = spawn(
      'scp',
      [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', localPath, target],
      {
        stdio: 'inherit',
      },
    );
    proc.on('error', reject);
    proc.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`scp exited with code ${code}`));
    });
  });
}

/**
 * Tail a remote file continuously (tail -n +1 -f), streaming lines to
 * process.stdout with an optional prefix. Returns a handle to stop tailing.
 * @param config
 * @param remotePath
 * @param prefix
 */
function tailRemoteLog(
  config: Credentials,
  remotePath: string,
  prefix = '',
): { stop: () => void } {
  const target = `${config.username}@${config.loginHost}`;
  const proc = spawn(
    'ssh',
    [
      ...SSH_MUX_OPTS,
      '-o',
      'BatchMode=yes',
      target,
      `tail -n +1 -f ${remotePath} 2>/dev/null`,
    ],
    { stdio: ['ignore', 'pipe', 'ignore'] },
  );

  let buf = '';
  proc.stdout?.on('data', (chunk: Buffer) => {
    buf += chunk.toString();
    const lines = buf.split('\n');
    buf = lines.pop() ?? '';
    for (const line of lines) {
      process.stdout.write(prefix + line + '\n');
    }
  });

  return {
    stop: () => {
      try {
        proc.kill();
      } catch {
        /* ignore */
      }
    },
  };
}

/**
 * Spawn a persistent forward SSH tunnel as a background child process.
 * ssh -N -L localPort:remoteHost:remotePort user@loginHost
 * Returns the child process so the caller can kill it on shutdown.
 * @param config
 * @param localPort
 * @param remoteHost typically the compute node
 * @param remotePort
 */
function spawnTunnel(
  config: Credentials,
  localPort: number,
  remoteHost: string,
  remotePort: number,
): CloseableEventEmitter {
  const target = `${config.username}@${config.loginHost}`;
  const proc = spawn(
    'ssh',
    [
      '-N',
      ...SSH_MUX_OPTS,
      '-o',
      'BatchMode=yes',
      '-o',
      'ServerAliveInterval=10',
      '-o',
      'ServerAliveCountMax=3',
      '-o',
      'ExitOnForwardFailure=yes',
      '-L',
      `${localPort}:${remoteHost}:${remotePort}`,
      target,
    ],
    { stdio: 'ignore', detached: false },
  );

  return proc;
}

/**
 *
 * @param config
 * @param ops
 */
async function listInstalledVersions(config: Credentials): Promise<string[]> {
  const { stdout } = await runRemote(
    config,
    `ls -d ${config.projectDir}/ivllm/*/bin 2>/dev/null | sed 's|.*/ivllm/||; s|/bin||'`,
  );
  return stdout
    .trim()
    .split('\n')
    .filter((v) => v && /^\d+\.\d+/.test(v));
}

export async function matchVllmVersion(
  config: Credentials,
  minVllmVersion: string,
): Promise<string> {
  const installed = await listInstalledVersions(config);
  if (installed.length === 0) {
    throw new Error(
      `No vLLM installation found at ${config.projectDir}/ivllm/. Run 'ivllm setup <version>'.`,
    );
  }

  const minVersion = minVllmVersion;
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

  return bestVersion;
}

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

export async function checkSSH(credentials: Credentials) {
  console.log('Checking SSH connectivity...');
  const { exitCode: sshCheck } = await runRemote(credentials, 'echo ok');
  if (sshCheck !== 0) {
    console.error('Error: Cannot connect to login node.');
    process.exit(1);
  }
  console.log('✓ SSH connectivity OK');
}

import { spawn } from 'child_process';

import readline from 'readline';
import type { Config } from './config.ts';

// Reusable SSH multiplexing options. The first connection spawns a
// background ControlMaster; subsequent connections within 10 minutes
// reuse the existing socket, avoiding repeated handshakes and login
// rate-limits on busy HPC login nodes.
const SSH_MUX_OPTS = [
  '-o', 'ControlMaster=auto',
  '-o', 'ControlPersist=600',
  '-o', 'ControlPath=/tmp/ivllm-ssh-%r@%h:%p',
] as const;
/**
 * Run a command on the LOGIN node via SSH, streaming stdout/stderr to the
 * current terminal. Returns a promise that resolves with the exit code.
 * @param config
 * @param command
 * @param options
 * @param options.env
 * @param options.silent
 */
export function runRemote(
  config: Config,
  command: string,
  options: { env?: Record<string, string>; silent?: boolean } = {},
): Promise<{ exitCode: number; stdout: string }> {
  return new Promise((resolve, reject) => {
    const target = `${config.username}@${config.loginHost}`;
    const envPrefix = options.env
      ? Object.entries(options.env)
          .map(([k, v]) => `${k}=${v}`)
          .join(' ') + ' '
      : '';
    const fullCommand = envPrefix + command;

    const proc = spawn('ssh', [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, fullCommand], {
      stdio: options.silent
        ? ['ignore', 'pipe', 'pipe']
        : ['ignore', 'inherit', 'inherit'],
    });

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

export function streamSrun(
  config: Config,
  command: string,
  options: {
    env?: Record<string, string>;
    silent?: boolean;
    onIdReceived?: (id: string) => void; // Dispatched instantly when found
  } = {},
): Promise<{ exitCode: number }> {
  const cmd = `${command} & echo "SLURM_JOB_ID_MATCH:$!"`;
  return new Promise((resolve, reject) => {
    const target = `${config.username}@${config.loginHost}`;
    const envPrefix = options.env
      ? Object.entries(options.env)
          .map(([k, v]) => `${k}=${v}`)
          .join(' ') + ' '
      : '';
    const fullCommand = envPrefix + cmd;

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

    let idReceived = false;
    rl.on('line', (line) => {
      // Print to local console unless silent
      if (!options.silent) {
        console.log(line);
      }

      // Extremely robust matching of SLURM identifier formats
      if (options.onIdReceived && !idReceived) {
        let foundId: string | null = null;
        const match = line.match(/Submitted (batch|interactive) job (\d+)/);
        if (match) {
          foundId = match[2];
        } else {
          const srunMatch = line.match(/srun: (?:job|Job) (\d+)/i);
          if (srunMatch) {
            foundId = srunMatch[1];
          } else if (line.includes('SLURM_JOB_ID_MATCH:')) {
            const pidMatch = line.match(/SLURM_JOB_ID_MATCH:(\d+)/);
            if (pidMatch) {
              foundId = pidMatch[1];
            }
          } else if (line.trim().match(/^\d+$/)) {
            foundId = line.trim();
          }
        }

        if (foundId) {
          idReceived = true;
          options.onIdReceived(foundId);
        }
      }
    });

    proc.stderr?.on('data', (chunk: Buffer) => {
      if (!options.silent) {
        process.stderr.write(chunk.toString());
      }
    });

    proc.on('error', reject);
    proc.on('close', (code) => {
      resolve({ exitCode: code ?? 1 });
    });
  });
}

/**
 * Copy a local file to a path on the LOGIN node via scp.
 * Returns a promise that resolves when the copy is complete.
 * @param config
 * @param localPath
 * @param remotePath
 */
export function copyFile(
  config: Config,
  localPath: string,
  remotePath: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const target = `${config.username}@${config.loginHost}:${remotePath}`;
    const proc = spawn('scp', [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', localPath, target], {
      stdio: 'inherit',
    });
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
export function tailRemoteLog(
  config: Config,
  remotePath: string,
  prefix = '',
): { stop: () => void } {
  const target = `${config.username}@${config.loginHost}`;
  const proc = spawn(
    'ssh',
    [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, `tail -n +1 -f ${remotePath} 2>/dev/null`],
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
 * @param remoteHost
 * @param remotePort
 */
export function spawnTunnel(
  config: Config,
  localPort: number,
  remoteHost: string,
  remotePort: number,
) {
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

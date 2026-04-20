import { spawn } from "child_process";
import type { Config } from "./config.ts";

/**
 * Run a command on the LOGIN node via SSH, streaming stdout/stderr to the
 * current terminal. Returns a promise that resolves with the exit code.
 */
export function runRemote(
  config: Config,
  command: string,
  options: { env?: Record<string, string>; silent?: boolean } = {}
): Promise<{ exitCode: number; stdout: string }> {
  return new Promise((resolve, reject) => {
    const target = `${config.username}@${config.loginHost}`;
    const envPrefix = options.env
      ? Object.entries(options.env)
          .map(([k, v]) => `${k}=${v}`)
          .join(" ") + " "
      : "";
    const fullCommand = envPrefix + command;

    const proc = spawn("ssh", ["-o", "BatchMode=yes", target, fullCommand], {
      stdio: options.silent ? ["ignore", "pipe", "pipe"] : ["ignore", "inherit", "inherit"],
    });

    let stdout = "";
    if (options.silent) {
      proc.stdout?.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
    }

    proc.on("error", reject);
    proc.on("close", (code) => resolve({ exitCode: code ?? 1, stdout: stdout.trim() }));
  });
}

/**
 * Copy a local file to a path on the LOGIN node via scp.
 * Returns a promise that resolves when the copy is complete.
 */
export function copyFile(
  config: Config,
  localPath: string,
  remotePath: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const target = `${config.username}@${config.loginHost}:${remotePath}`;
    const proc = spawn("scp", ["-o", "BatchMode=yes", localPath, target], {
      stdio: "inherit",
    });
    proc.on("error", reject);
    proc.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`scp exited with code ${code}`));
    });
  });
}

/**
 * Tail a remote file continuously (tail -n +1 -f), streaming lines to
 * process.stdout with an optional prefix. Returns a handle to stop tailing.
 */
export function tailRemoteLog(
  config: Config,
  remotePath: string,
  prefix = ""
): { stop: () => void } {
  const target = `${config.username}@${config.loginHost}`;
  const proc = spawn(
    "ssh",
    ["-o", "BatchMode=yes", target, `tail -n +1 -f ${remotePath} 2>/dev/null`],
    { stdio: ["ignore", "pipe", "ignore"] }
  );

  let buf = "";
  proc.stdout?.on("data", (chunk: Buffer) => {
    buf += chunk.toString();
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      process.stdout.write(prefix + line + "\n");
    }
  });

  return {
    stop: () => {
      try { proc.kill(); } catch { /* ignore */ }
    },
  };
}

/**
 * Spawn a persistent forward SSH tunnel as a background child process.
 * ssh -N -L localPort:remoteHost:remotePort user@loginHost
 * Returns the child process so the caller can kill it on shutdown.
 */
export function spawnTunnel(
  config: Config,
  localPort: number,
  remoteHost: string,
  remotePort: number
) {
  const target = `${config.username}@${config.loginHost}`;
  const proc = spawn(
    "ssh",
    [
      "-N",
      "-o", "BatchMode=yes",
      "-o", "ServerAliveInterval=10",
      "-o", "ServerAliveCountMax=3",
      "-o", "ExitOnForwardFailure=yes",
      "-L", `${localPort}:${remoteHost}:${remotePort}`,
      target,
    ],
    { stdio: "ignore", detached: false }
  );
  return proc;
}

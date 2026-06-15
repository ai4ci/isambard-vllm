import { type Config } from "../config.ts";
import { makeRemoteOps } from "../remote-ops.ts";
import { semverGte, semverSort } from "./semver.ts";

/**
 * Given a list of installed vLLM versions and a minimum required version,
 * returns the highest installed version that satisfies the minimum, or null if none do.
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
 * Lists installed vLLM versions by scanning $PROJECTDIR/ivllm/ for versioned venv directories.
 * Returns versions that have a bin/ directory (i.e. are complete installs).
 */
export async function listInstalledVersions(
  config: Config,
  ops: ReturnType<typeof makeRemoteOps>,
): Promise<string[]> {
  const { stdout } = await ops.runRemote(
    `ls -d ${config.projectDir}/ivllm/*/bin 2>/dev/null | sed 's|.*/ivllm/||; s|/bin||'`,
    { silent: true },
  );
  return stdout
    .trim()
    .split('\n')
    .filter((v) => v && /^\d+\.\d+/.test(v));
}

export function timestamp(): string {
  return new Date().toTimeString().slice(0, 8);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Check if a local TCP port is already in use.
 * Returns `{ pid, process }` if occupied, or `null` if free.
 * Uses `lsof` (macOS/Linux) with fallback to `/proc/net/tcp`.
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
      const pid = stdout.trim().split('\n')[0];
      // Try to get the process name for a nicer error message
      execFile('ps', ['-p', pid, '-o', 'comm='], (_err2, psOut) => {
        const process = psOut?.trim() || 'unknown';
        resolve({ pid, process });
      });
    });
  });
}

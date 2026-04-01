import { copyFileSync } from "fs";
import { basename, join } from "path";
import type { Config } from "./config.ts";
import { runRemote as _runRemote, copyFile as _copyFile } from "./ssh.ts";

type RunRemoteOptions = { env?: Record<string, string>; silent?: boolean };
type RunRemoteResult = { exitCode: number; stdout: string };

export interface RemoteOps {
  runRemote(command: string, options?: RunRemoteOptions): Promise<RunRemoteResult>;
  copyFile(localPath: string, remotePath: string): Promise<void>;
}

/**
 * Returns real RemoteOps that execute SSH/SCP, or dry-run ops that print
 * what would happen and copy files to a local preview directory.
 */
export function makeRemoteOps(config: Config, dryRun: boolean, dryRunDir?: string): RemoteOps {
  if (!dryRun) {
    return {
      runRemote: (cmd, opts) => _runRemote(config, cmd, opts),
      copyFile: (local, remote) => _copyFile(config, local, remote),
    };
  }

  return {
    async runRemote(command, opts) {
      if (!opts?.silent) {
        console.log(`  [dry-run] Would run remotely:\n    ${command}`);
      }
      return { exitCode: 0, stdout: "" };
    },
    async copyFile(localPath, remotePath) {
      const destName = basename(remotePath);
      const dest = join(dryRunDir!, destName);
      copyFileSync(localPath, dest);
      console.log(`  [dry-run] Would scp: ${localPath} → ${remotePath}`);
      console.log(`           (preview: ${dest})`);
    },
  };
}

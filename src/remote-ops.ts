import { copyFileSync, mkdtempSync } from 'fs';
import { basename, join } from 'path';
import type { Config, RemoteOps, StartArgs } from './types.ts';
import {
  runRemote as _runRemote,
  copyFile as _copyFile,
  spawnTunnel as _spawnTunnel,
  streamSrun as _streamSrun,
  tailRemoteLog as _tailRemoteLog
} from './ssh.ts';
import EventEmitter from 'node:events';


/**
 * Returns real RemoteOps that execute SSH/SCP, or dry-run ops that print
 * what would happen and copy files to a local preview directory.
 * @param config
 * @param dryRun
 * @param dryRunDir
 */
export function makeRemoteOps(
  config: Config,
  startArgs: StartArgs
): RemoteOps {
  if (!startArgs.dryRun) {

    return {
      runRemote: (cmd, opts) => _runRemote(config, cmd, opts),
      copyFile: (local, remote) => _copyFile(config, local, remote),
      streamSrun: (cmd, opts) => _streamSrun(config, cmd, opts),
      tailRemoteLog: (remote) => _tailRemoteLog(config, remote),
      spawnTunnel:(local, remoteHost, remotePort) => _spawnTunnel(config, local, remoteHost, remotePort),
    };
  }

  return {

    async runRemote(command, opts) {
      console.log(`  [dry-run] Would run remotely:\n    ${command}`);
      return { exitCode: 0, stdout: 'Submitted batch job 123456' };
    },
    async copyFile(localPath, remotePath) {
      const dryRunDir = mkdtempSync(join(process.cwd(), 'ivllm-dryrun-'));
      const destName = basename(remotePath);
      const dest = join(dryRunDir!, destName);
      copyFileSync(localPath, dest);
      console.log(`  [dry-run] Would scp: ${localPath} → ${remotePath}`);
      console.log(`           (preview: ${dest})`);
    },
    async streamSrun(command, opts) {
      console.log(`  [dry-run] Would stream remotely:\n    ${command}`);
      return { exitCode: 0, stdout: '' };
    },
    tailRemoteLog(remote) {
      console.log(`dummy log entry from ${remote}`);
      return { stop: () => void };
    },
    spawnTunnel(local, remoteHost, remotePort) {
      console.log(`Mock SSH tunnel created with ${local}:${remoteHost}:${remotePort}`);

      // 1. Create the main process and its standard I/O stubs as plain EventEmitters
      const mockSshTunnel = Object.assign(new EventEmitter(), {
        stdin: Object.assign(
          new EventEmitter(), { write: () => true }
        ),
        stdout: Object.assign(new EventEmitter(), {
          pipe: (destination: EventEmitter) => destination
        }),
        stderr: Object.assign(new EventEmitter(),  {
          pipe: (destination: EventEmitter) => destination
        }),
        // 2. Add required identity properties so it looks like a running process
        pid = 99999,
        exitCode = null, // Stays null so it looks permanently alive
        kill = () => {
          mockSshTunnel.emit('close', 0);
          return true;
        }
      });

      return mockSshTunnel;


    }
  };
}

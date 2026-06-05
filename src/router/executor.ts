import { RemoteExecutor } from '../types.js';

/**
 * SSH-based remote executor for Isambard AI
 */
export class SSHExecutor implements RemoteExecutor {
  private host: string;

  constructor(host: string) {
    this.host = host;
  }

  async runCommand(command: string): Promise<string> {
    const { exec } = await import('child_process');
    
    return new Promise((resolve, reject) => {
      exec(`ssh ${this.host} "${command}"`, (error, stdout, stderr) => {
        if (error) {
          reject(new Error(`SSH command failed: ${stderr || error.message}`));
          return;
        }
        resolve(stdout.trim());
      });
    });
  }

  async copyFile(localPath: string, remotePath: string): Promise<void> {
    const { exec } = await import('child_process');
    
    return new Promise((resolve, reject) => {
      exec(`scp ${localPath} ${this.host}:${remotePath}`, (error, stdout, stderr) => {
        if (error) {
          reject(new Error(`SCP failed: ${stderr || error.message}`));
          return;
        }
        resolve();
      });
    });
  }

  async readFile(remotePath: string): Promise<string> {
    return this.runCommand(`cat ${remotePath}`);
  }

  async fileExists(remotePath: string): Promise<boolean> {
    try {
      await this.runCommand(`test -f ${remotePath} && echo "exists"`);
      return true;
    } catch {
      return false;
    }
  }
}

/**
 * Local executor for login node mode
 */
export class LocalExecutor implements RemoteExecutor {
  async runCommand(command: string): Promise<string> {
    const { exec } = await import('child_process');
    
    return new Promise((resolve, reject) => {
      exec(command, (error, stdout, stderr) => {
        if (error) {
          reject(new Error(`Command failed: ${stderr || error.message}`));
          return;
        }
        resolve(stdout.trim());
      });
    });
  }

  async copyFile(localPath: string, remotePath: string): Promise<void> {
    const fs = await import('fs');
    await fs.promises.copyFile(localPath, remotePath);
  }

  async readFile(remotePath: string): Promise<string> {
    const fs = await import('fs');
    return fs.promises.readFile(remotePath, 'utf-8');
  }

  async fileExists(remotePath: string): Promise<boolean> {
    const fs = await import('fs');
    try {
      await fs.promises.access(remotePath);
      return true;
    } catch {
      return false;
    }
  }
}

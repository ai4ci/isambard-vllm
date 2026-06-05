/**
 * Remote executor interface - abstracts SSH vs local execution
 */
export interface RemoteExecutor {
  runCommand(command: string): Promise<string>;
  copyFile(localPath: string, remotePath: string): Promise<void>;
  readFile(remotePath: string): Promise<string>;
  fileExists(remotePath: string): Promise<boolean>;
}

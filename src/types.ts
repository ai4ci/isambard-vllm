import type EventEmitter from 'events';

// =================================
// CONFIG AND CMD LINE OPTIONS
// =================================

export interface Credentials {
  loginHost: string;
  username: string;
  projectDir: string;
  defaultLocalPort: number;
  hfToken?: string;
}

export interface InferenceJobOptions {
  jobName: string;
  credentials: Credentials;
  configFile: string; // local vllm.yaml path
  configYaml: ServeOptions; // local parsed vllm config
  localPort: number;
  gpuCount: number; // if unset, derived from tensor-parallel-size in YAML in start.ts
  timeLimit: string;
  serverPort: number;
  mock: boolean;
  dryRun: boolean;
  noLaunch: boolean; // skip assistant launch even in non-dryRun mode
  isInteractive: boolean; // launch with srun in interactive cluster
  preCache: boolean; // pre cache compilation using slurm batch and exit model when running.
  cacheKey: string; //key to use for compiled models
}

export interface SimplePaths {
  remoteProjectDir: string;
  remoteHomeDir: string;
  remoteProjectVllmDir: string;
  remoteProjectVllmPluginsDir: string;
  remoteProjectVllmVersionDir: string;
  remoteProjectVllmVenvActivate: string;
  nvhpcDir: string;
  nvhpcRoot: string;
}

export interface Paths extends SimplePaths {
  remoteJobDir: string;
  remoteJobLockFile: string;
  remoteJobVllmConfigFile: string;
  remoteJobVllmPluginsDir: string;
  remoteJobScriptFile: string;
  remoteJobLogFile: string;
  remoteProjectHfDir: string;
  remoteProjectHfModelDir: string;
  remoteProjectJobCacheDir: string;
  remoteProjectJobCacheFile: string;
  localCacheDir: string;
  localCacheVllmConfigFile: string;
}

export interface ServeOptions {
  model: string;
  tensorParallelSize: number;
  pipelineParallelSize: number;
  dataParallelSize: number;
  maxModelLen: number;
  enableAutoToolChoice: boolean;
  enableReasoning: boolean;
  minVllmVersion: string;
  /** Environment variables to set before launching vLLM. Always present (may be empty). */
  env: EnvVarEntry[];
  raw: Record<string, unknown>;
}

/** Parsed environment variable entry. */
export interface EnvVarEntry {
  key: string;
  value: string;
}

export class ProcessState {
  sessionName!: string;
  ops!: RemoteOps;
  paths!: SimplePaths;
  vllmVersion!: string; // the version that is to be used.
  remoteCommand?: string;
  slurmJobId?: string;
  process?: CloseableEventEmitter;
  tunnel?: CloseableEventEmitter;
  heartbeatTimer?: Timer;
  crashDiagnosticsPrinted?: boolean;
  shuttingDown?: boolean;

  constructor(init?: Partial<ProcessState>) {
    if (init) Object.assign(this, init);
  }
}

export class SessionState extends ProcessState {
  declare paths: Paths;
  localOps!: LocalOps;
  startArgs!: InferenceJobOptions;

  constructor(init?: Partial<SessionState>) {
    super();
    if (init) Object.assign(this, init);
  }
}

// =================================
// JOB CONFIGURATION OPTIONS
// =================================

export type JobStatus =
  | 'pending'
  | 'initialising'
  | 'running'
  | 'failed'
  | 'timeout';

/** Metadata for a stored job config. */
export interface JobConfigEntry {
  jobName: string;
  filePath: string;
  model?: string;
  tensorParallelSize?: number;
  pipelineParallelSize?: number;
}

/** Remote metadata / lockfile stored on HPC **/
export interface JobDetails {
  status: JobStatus;
  job_name: string;
  slurm_job_id?: string;
  compute_hostname?: string;
  model?: string;
  server_port?: number;
  error?: string;
}

// =================================
// REMOTE OPERATIONS INTERFACE
// =================================

export type RunRemoteOptions = {
  env: EnvVarEntry[];
  silent?: boolean;
};

export type RunRemoteResult = {
  exitCode: number;
  stdout: string;
};

export interface RemoteOps {
  runRemote(
    command: string,
    options?: RunRemoteOptions,
  ): Promise<RunRemoteResult>;
  streamSrun(
    command: string,
    sessionState: ProcessState,
    options?: RunRemoteOptions,
  ): CloseableEventEmitter;
  copyFile(localPath: string, remotePath: string): Promise<void>;
  tailRemoteLog(remotePath: string, prefix?: string): { stop: () => void };
  spawnTunnel(
    localPort: number,
    remoteHost: string,
    remotePort: number,
  ): CloseableEventEmitter;
  matchVllmVersion(minVllmVersion: string): Promise<string>;
  checkSSH(): Promise<boolean>;
}

export interface LocalOps {
  checkLocalHealth(localPort: number): Promise<boolean>;
  queryModels(localPort: number): Promise<V1ModelsResponse>;
  isLocalPortInUse(
    localPort: number,
  ): Promise<{ pid: string; process: string } | null>;
}

// Blocks unit finishes or not as the case may be,
// Clean up and wait for target to resolve maybe.
export interface RemoteMonitor {
  start(state: ProcessState): Promise<void>;
}

export interface CloseableEventEmitter extends EventEmitter {
  kill(signal?: NodeJS.Signals | number): boolean;
}

// =================================
// VLLM API
// =================================

export interface V1ModelsResponse {
  object: 'list';
  data: Array<{
    id: string;
    [key: string]: any;
    max_model_len?: number;
  }>;
}

// =================================

// export interface InferenceScriptOptions {
//     jobName: string;
//     model: string;
//     vllmVersion: string;
//     hfHome: string;
//     configFileName: string;
//     workDir: string;
//     serverPort: number;
//     gpuCount: number;
//     nodeCount: number;
//     timeLimit: string;
//     envVars: EnvVarEntry[];
//     isInteractive: boolean;
//     cacheKey: string;
//     preCache: boolean;
// }

// export interface MonitorRuntimeOpts {
//     localPort: number;
//     serverPort: number;
//     config: Credentials;
//     model: string;
//     maxModelLen?: number;
//     enableAutoToolChoice: boolean;
//     enableReasoning: boolean;
//     isInteractive: boolean;
// }

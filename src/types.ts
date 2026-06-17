import type EventEmitter from 'events';

// =================================
// CONFIG AND CMD LINE OPTIONS
// =================================

class Test {
    value: string = "default";
    public print(): string {return this.value;}
}

const tmp = new Test();

export interface Config {
    loginHost: string;
    username: string;
    projectDir: string;
    defaultLocalPort: number;
    hfToken?: string;
}

export interface StartArgs {
    jobName: string;
    model?: string; // only populated for --mock mode; non-mock reads model from YAML
    configFile: string; // required unless mock: true
    configYaml: VllmConfig;
    localPort?: number;
    gpuCount?: number; // if unset, derived from tensor-parallel-size in YAML in start.ts
    timeLimit: string;
    serverPort: number;
    mock: boolean;
    dryRun: boolean;
    noLaunch: boolean; // skip assistant launch even in non-dryRun mode
    preCache: boolean; // pre cache compilation using slurm batch and exit model when running.
}

export interface VllmConfig {
    model?: string;
    tensorParallelSize?: number;
    pipelineParallelSize?: number;
    maxModelLen?: number;
    enableAutoToolChoice?: boolean;
    enableReasoning?: boolean;
    minVllmVersion?: string;
    /** Environment variables to set before launching vLLM. Always present (may be empty). */
    env: EnvVarEntry[];
    raw: Record<string, unknown>;
}

/** Parsed environment variable entry. */
export interface EnvVarEntry {
    key: string;
    value: string;
}

// =================================
// JOB CONFIGURATION OPTIONS
// =================================

export type JobStatus = 'pending' | 'initialising' | 'running' | 'failed' | 'timeout';

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
    silent?: boolean
};

export type RunRemoteResult = {
    exitCode: number;
    stdout: string
};

export interface RemoteOps {
    runRemote(
        command: string,
        options?: RunRemoteOptions,
    ): Promise<RunRemoteResult>;
    streamSrun(
        command: string,
        options?: RunRemoteOptions
    ): Promise<RunRemoteResult>;
    copyFile(
        localPath: string,
        remotePath: string
    ): Promise<void>;
    tailRemoteLog(
        remotePath: string
    ): { stop: () => void };
    spawnTunnel(
        localPort: number,
        remoteHost: string,
        remotePort: number): CloseableEventEmitter
}

export interface RemoteMonitor {
    start(
        state: SessionState,
        args: StartArgs,
        runtimeOpts: MonitorRuntimeOpts,
    ): Promise<void>,
}

export interface CloseableEventEmitter extends EventEmitter {
    kill(signal?: NodeJS.Signals | number): boolean;
}

// =================================


export interface InferenceScriptOptions {
    jobName: string;
    model: string;
    vllmVersion: string;
    hfHome: string;
    configFileName: string;
    workDir: string;
    serverPort: number;
    gpuCount: number;
    nodeCount: number;
    timeLimit: string;
    envVars: EnvVarEntry[];
    isInteractive: boolean;
    cacheKey: string;
    preCache: boolean;
}



export interface SessionState {
    jobName: string;
    config: Config;
    ops: RemoteOps;
    remoteWorkDir: string;
    remoteJobDetails: string;
    slurmJobId: string | null;
    tunnel: EventEmitter | null;
    heartbeatTimer: ReturnType<typeof setInterval> | null;
    crashDiagnosticsPrinted: boolean;
    shuttingDown: boolean;
}

export interface ResolvedSessionConfig {
    configFile: string;
    usingStoredConfig: boolean;
    yamlConfig: VllmConfig;
    model: string;
    gpuCount: number;
    nodeCount: number;
    maxModelLen?: number;
    tensorParallelSize: number;
    pipelineParallelSize: number;
    enableAutoToolChoice: boolean;
    enableReasoning: boolean;
    vllmVersion: string;
    envVars: EnvVarEntry[];
    preCache: boolean;
}

export interface MonitorRuntimeOpts {
    localPort: number;
    serverPort: number;
    config: Config;
    model: string;
    maxModelLen?: number;
    enableAutoToolChoice: boolean;
    enableReasoning: boolean;
    isInteractive: boolean;
}

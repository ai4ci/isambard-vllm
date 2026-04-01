export type JobStatus = "pending" | "initialising" | "running" | "failed" | "timeout";

export interface JobDetails {
  status: JobStatus;
  job_name: string;
  slurm_job_id?: string;
  compute_hostname?: string;
  model?: string;
  server_port?: number;
  error?: string;
}

export interface StartArgs {
  jobName: string;
  model: string;
  configFile: string;
  localPort?: number;
  gpuCount: number;
  tensorParallelSize: number;
  timeLimit: string;
  serverPort: number;
}

export function parseJobDetails(raw: string): JobDetails | null {
  if (!raw.trim()) return null;
  try {
    const obj = JSON.parse(raw) as Record<string, unknown>;
    if (typeof obj["status"] !== "string") return null;
    return obj as unknown as JobDetails;
  } catch {
    return null;
  }
}

export function hfCachePath(projectDir: string, model: string): string {
  const cacheKey = model.includes("/")
    ? "models--" + model.replace("/", "--")
    : "models--" + model;
  return `${projectDir}/hub/${cacheKey}`;
}

export function parseStartArgs(args: string[]): StartArgs {
  // First positional arg is job name — it must not start with --
  const jobName = args[0] && !args[0].startsWith("--") ? args[0] : null;
  if (!jobName) throw new Error("Job name is required as the first argument");

  const flags: Record<string, string> = {};
  for (let i = 1; i < args.length - 1; i++) {
    const arg = args[i];
    if (arg?.startsWith("--")) flags[arg.slice(2)] = args[i + 1] ?? "";
  }

  if (!flags["model"]) throw new Error("--model <model> is required");
  if (!flags["config"]) throw new Error("--config <file> is required");

  const gpuCount = flags["gpus"] ? parseInt(flags["gpus"], 10) : 4;
  const tensorParallelSize = flags["tensor-parallel-size"]
    ? parseInt(flags["tensor-parallel-size"], 10)
    : gpuCount;

  return {
    jobName,
    model: flags["model"]!,
    configFile: flags["config"]!,
    localPort: flags["local-port"] ? parseInt(flags["local-port"], 10) : undefined,
    gpuCount,
    tensorParallelSize,
    timeLimit: flags["time"] ?? "4:00:00",
    serverPort: flags["server-port"] ? parseInt(flags["server-port"], 10) : 8000,
  };
}

import type { JobDetails, StartArgs } from "./types";
import { jobConfigPath, parseVllmConfig } from "./vllm-config";


/**
 *
 * @param raw
 */
export function parseJobDetails(raw: string): JobDetails | null {
  if (!raw.trim()) return null;
  try {
    const obj = JSON.parse(raw) as Record<string, unknown>;
    if (typeof obj['status'] !== 'string') return null;
    return obj as unknown as JobDetails;
  } catch {
    return null;
  }
}

/**
 *
 * @param projectDir
 * @param model
 */
export function hfCachePath(projectDir: string, model: string): string {
  const cacheKey = model.includes('/')
    ? 'models--' + model.replace('/', '--')
    : 'models--' + model;
  return `${projectDir}/hub/${cacheKey}`;
}

/**
 *
 * @param args
 */
export function parseStartArgs(args: string[]): StartArgs {
  // First positional arg is job name — it must not start with --
  const jobName = args[0] && !args[0].startsWith('--') ? args[0] : null;
  if (!jobName) throw new Error('Job name is required as the first argument');

  // Parse boolean flags and key=value flags
  const boolFlags = new Set<string>();
  const flags: Record<string, string> = {};
  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (!arg?.startsWith('--')) continue;
    const key = arg.slice(2);
    const next = args[i + 1];
    if (!next || next.startsWith('--')) {
      boolFlags.add(key);
    } else {
      flags[key] = next;
      i++;
    }
  }

  const mock = boolFlags.has('mock');
  const dryRun = boolFlags.has('dry-run');
  const noLaunch = boolFlags.has('no-launch');
  const preCache = boolFlags.has('create-cache');

  if (mock && !flags['model'])
    throw new Error('--model <model> is required in mock mode');

  const gpuCount = flags['gpus'] ? parseInt(flags['gpus'], 10) : undefined;

  let configPath = flags['config'] ?? jobConfigPath(jobName);
  let yaml = parseVllmConfig(configPath);

  return {
    jobName,
    model: flags['model'],
    configFile: configPath,
    configYaml: yaml,
    localPort: flags['local-port']
      ? parseInt(flags['local-port'], 10)
      : undefined,
    gpuCount,
    timeLimit: flags['time'] ?? '8:00:00',
    serverPort: flags['server-port']
      ? parseInt(flags['server-port'], 10)
      : 8000,
    mock,
    dryRun,
    noLaunch,
    preCache,
  };
}

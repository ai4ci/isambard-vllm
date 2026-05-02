import { readFileSync, writeFileSync, mkdirSync, existsSync, copyFileSync } from "fs";
import { join } from "path";
import { tmpdir, homedir } from "os";
import yaml from "js-yaml";

/** Keys that are ivllm-specific and must be stripped before passing the config to `vllm serve`. */
export const IVLLM_ONLY_KEYS = new Set(["min-vllm-version"]);

const JOB_CONFIG_DIR = join(homedir(), ".config", "ivllm");

/** Returns the path where a named job's vLLM config is stored locally. */
export function jobConfigPath(jobName: string): string {
  return join(JOB_CONFIG_DIR, `${jobName}.yaml`);
}

/** Saves a copy of the given vLLM config file to the local job config store. */
export function saveJobConfig(jobName: string, sourcePath: string): void {
  if (!existsSync(JOB_CONFIG_DIR)) {
    mkdirSync(JOB_CONFIG_DIR, { recursive: true });
  }
  copyFileSync(sourcePath, jobConfigPath(jobName));
}

export interface VllmConfig {
  model?: string;
  tensorParallelSize?: number;
  pipelineParallelSize?: number;
  maxModelLen?: number;
  enableAutoToolChoice?: boolean;
  enableReasoning?: boolean;
  minVllmVersion?: string;
}

/**
 * Resolve the total GPU count and node count for a SLURM job from a vLLM config.
 * gpus = tensor_parallel_size * pipeline_parallel_size.
 * nodeCount = ceil(gpuCount / gpusPerNode).
 * If --gpus was explicitly set on the CLI, that takes precedence for gpuCount.
 */
export function resolveGpuCount(
  cliGpus: number | undefined,
  yamlConfig: VllmConfig,
  gpusPerNode = 4
): { gpuCount: number; nodeCount: number; error?: string } {
  if (cliGpus !== undefined) {
    return { gpuCount: cliGpus, nodeCount: Math.ceil(cliGpus / gpusPerNode) };
  }

  const tp = yamlConfig.tensorParallelSize ?? 4;
  const pp = yamlConfig.pipelineParallelSize ?? 1;
  const gpuCount = tp * pp;
  const nodeCount = Math.ceil(gpuCount / gpusPerNode);

  return { gpuCount, nodeCount };
}

/**
 * Parse a vLLM YAML config file and extract fields that ivllm needs locally
 * (for HF model download and SLURM GPU allocation).
 */
export function parseVllmConfig(filePath: string): VllmConfig {
  const raw = readFileSync(filePath, "utf-8");
  const doc = yaml.load(raw) as Record<string, unknown>;

  const model = typeof doc["model"] === "string" ? doc["model"] : undefined;

  // vLLM YAML uses kebab-case keys; also accept underscore variant
  const tp = doc["tensor-parallel-size"] ?? doc["tensor_parallel_size"];
  const tensorParallelSize = typeof tp === "number" ? tp : undefined;

  // vLLM YAML uses kebab-case keys; also accept underscore variant
  const pp = doc["pipeline-parallel-size"] ?? doc["pipeline_parallel_size"];
  const pipelineParallelSize = typeof pp === "number" ? pp : undefined;

  const minVer = doc["min-vllm-version"];
  const minVllmVersion = typeof minVer === "string" ? minVer : undefined;

  const mml = doc["max-model-len"] ?? doc["max_model_len"];
  const maxModelLen = typeof mml === "number" ? mml : undefined;

  const toolChoice = doc["enable-auto-tool-choice"] ?? doc["enable_auto_tool_choice"];
  const enableAutoToolChoice = typeof toolChoice === "boolean" ? toolChoice : undefined;

  // Derive enableReasoning from the presence of reasoning-parser (not from enable-reasoning,
  // which is an opencode.js concept and not a valid vLLM config key).
  const reasoningParser = doc["reasoning-parser"] ?? doc["reasoning_parser"];
  const enableReasoning = typeof reasoningParser === "string" && reasoningParser.length > 0 ? true : undefined;

  return { model, tensorParallelSize, pipelineParallelSize, maxModelLen, enableAutoToolChoice, enableReasoning, minVllmVersion };
}

/**
 * Returns a cleaned YAML string with all ivllm-specific keys removed.
 * vLLM errors on unknown config keys — always use this when uploading to the remote.
 */
export function stripIvllmKeys(filePath: string): string {
  const raw = readFileSync(filePath, "utf-8");
  const doc = yaml.load(raw) as Record<string, unknown>;
  for (const key of IVLLM_ONLY_KEYS) {
    delete doc[key];
  }
  return yaml.dump(doc, { lineWidth: -1 });
}

/**
 * Writes a stripped (ivllm-keys removed) copy of the YAML config to a temp file
 * and returns the temp file path. Caller is responsible for deleting it.
 */
export function writeStrippedConfig(filePath: string): string {
  const stripped = stripIvllmKeys(filePath);
  const tmpPath = join(tmpdir(), `ivllm-stripped-${Date.now()}.yaml`);
  writeFileSync(tmpPath, stripped, "utf-8");
  return tmpPath;
}

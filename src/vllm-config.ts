import { readFileSync } from "fs";
import yaml from "js-yaml";

export interface VllmConfig {
  model?: string;
  tensorParallelSize?: number;
  pipelineParallelSize?: number;
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

  return { model, tensorParallelSize, pipelineParallelSize };
}

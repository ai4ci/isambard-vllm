import { readFileSync } from "fs";
import yaml from "js-yaml";

export interface VllmConfig {
  model?: string;
  tensorParallelSize?: number;
  pipelineParallelSize?: number;
}

/**
 * Resolve the total GPU count for a SLURM job from a vLLM config.
 * For single-node jobs: gpus = tensor_parallel_size * pipeline_parallel_size.
 * If --gpus was explicitly set on the CLI, that takes precedence.
 *
 * Returns { gpuCount, error } where error is set if multi-node would be needed.
 */
export function resolveGpuCount(
  cliGpus: number | undefined,
  yamlConfig: VllmConfig,
  maxSingleNodeGpus = 4
): { gpuCount: number; error?: string } {
  if (cliGpus !== undefined) return { gpuCount: cliGpus };

  const tp = yamlConfig.tensorParallelSize ?? 4;
  const pp = yamlConfig.pipelineParallelSize ?? 1;
  const gpuCount = tp * pp;

  if (gpuCount > maxSingleNodeGpus) {
    return {
      gpuCount,
      error: `Product of tensor-parallel-size (${tp}) and pipeline-parallel-size (${pp}) = ${gpuCount}, which exceeds GPUs on a single node (${maxSingleNodeGpus}). Multi-node is not yet supported.`,
    };
  }

  return { gpuCount };
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

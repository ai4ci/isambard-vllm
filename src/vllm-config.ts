import { readFileSync } from "fs";
import yaml from "js-yaml";

export interface VllmConfig {
  model?: string;
  tensorParallelSize?: number;
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

  return { model, tensorParallelSize };
}

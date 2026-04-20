import { describe, it, expect } from "bun:test";
import { writeFileSync, unlinkSync, readFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { parseVllmConfig, resolveGpuCount, stripIvllmKeys, IVLLM_ONLY_KEYS } from "../src/vllm-config.ts";

function writeTmp(content: string): string {
  const path = join(tmpdir(), `ivllm-test-${Date.now()}.yaml`);
  writeFileSync(path, content, "utf-8");
  return path;
}

describe("parseVllmConfig", () => {
  it("extracts model from YAML", () => {
    const path = writeTmp("model: Qwen/Qwen2.5-0.5B-Instruct\n");
    try {
      expect(parseVllmConfig(path).model).toBe("Qwen/Qwen2.5-0.5B-Instruct");
    } finally { unlinkSync(path); }
  });

  it("extracts tensor-parallel-size (kebab-case) from YAML", () => {
    const path = writeTmp("tensor-parallel-size: 4\n");
    try {
      expect(parseVllmConfig(path).tensorParallelSize).toBe(4);
    } finally { unlinkSync(path); }
  });

  it("extracts tensor_parallel_size (underscore) from YAML", () => {
    const path = writeTmp("tensor_parallel_size: 8\n");
    try {
      expect(parseVllmConfig(path).tensorParallelSize).toBe(8);
    } finally { unlinkSync(path); }
  });

  it("extracts pipeline-parallel-size (kebab-case) from YAML", () => {
    const path = writeTmp("pipeline-parallel-size: 2\n");
    try {
      expect(parseVllmConfig(path).pipelineParallelSize).toBe(2);
    } finally { unlinkSync(path); }
  });

  it("extracts pipeline_parallel_size (underscore) from YAML", () => {
    const path = writeTmp("pipeline_parallel_size: 2\n");
    try {
      expect(parseVllmConfig(path).pipelineParallelSize).toBe(2);
    } finally { unlinkSync(path); }
  });

  it("returns undefined pipelineParallelSize when not present", () => {
    const path = writeTmp("model: some/model\n");
    try {
      expect(parseVllmConfig(path).pipelineParallelSize).toBeUndefined();
    } finally { unlinkSync(path); }
  });

  it("returns undefined model when not present", () => {
    const path = writeTmp("max-model-len: 8192\n");
    try {
      expect(parseVllmConfig(path).model).toBeUndefined();
    } finally { unlinkSync(path); }
  });

  it("returns undefined tensorParallelSize when not present", () => {
    const path = writeTmp("model: some/model\n");
    try {
      expect(parseVllmConfig(path).tensorParallelSize).toBeUndefined();
    } finally { unlinkSync(path); }
  });

  it("parses a realistic multi-line config", () => {
    const path = writeTmp(
      "model: Qwen/Qwen2.5-0.5B-Instruct\n" +
      "tensor-parallel-size: 4\n" +
      "max-model-len: 8192\n" +
      "gpu-memory-utilization: 0.90\n"
    );
    try {
      const result = parseVllmConfig(path);
      expect(result.model).toBe("Qwen/Qwen2.5-0.5B-Instruct");
      expect(result.tensorParallelSize).toBe(4);
    } finally { unlinkSync(path); }
  });

  it("throws a helpful error for a missing file", () => {
    expect(() => parseVllmConfig("/nonexistent/path.yaml")).toThrow();
  });

  it("throws a helpful error for invalid YAML", () => {
    const path = writeTmp("{{invalid: yaml: [\n");
    try {
      expect(() => parseVllmConfig(path)).toThrow();
    } finally { unlinkSync(path); }
  });

  it("extracts min-vllm-version (kebab-case) from YAML", () => {
    const path = writeTmp("min-vllm-version: \"0.9.1\"\n");
    try {
      expect(parseVllmConfig(path).minVllmVersion).toBe("0.9.1");
    } finally { unlinkSync(path); }
  });

  it("returns undefined minVllmVersion when not present", () => {
    const path = writeTmp("model: some/model\n");
    try {
      expect(parseVllmConfig(path).minVllmVersion).toBeUndefined();
    } finally { unlinkSync(path); }
  });
});

describe("resolveGpuCount", () => {
  it("returns CLI value when explicitly set, ignoring YAML", () => {
    expect(resolveGpuCount(2, { tensorParallelSize: 4, pipelineParallelSize: 2 }).gpuCount).toBe(2);
  });

  it("uses tensor-parallel-size from YAML when no CLI value", () => {
    expect(resolveGpuCount(undefined, { tensorParallelSize: 4 }).gpuCount).toBe(4);
  });

  it("multiplies tp and pp from YAML", () => {
    expect(resolveGpuCount(undefined, { tensorParallelSize: 2, pipelineParallelSize: 2 }).gpuCount).toBe(4);
  });

  it("defaults tp to 4 and pp to 1 when neither is in YAML", () => {
    expect(resolveGpuCount(undefined, {}).gpuCount).toBe(4);
  });

  it("returns no error when gpuCount is within single-node limit", () => {
    expect(resolveGpuCount(undefined, { tensorParallelSize: 4 }).error).toBeUndefined();
  });

  it("returns nodeCount=1 for single-node config (tp=4, pp=1)", () => {
    const result = resolveGpuCount(undefined, { tensorParallelSize: 4 });
    expect(result.nodeCount).toBe(1);
  });

  it("returns nodeCount=2 when tp * pp = 8 (two nodes)", () => {
    const result = resolveGpuCount(undefined, { tensorParallelSize: 4, pipelineParallelSize: 2 });
    expect(result.gpuCount).toBe(8);
    expect(result.nodeCount).toBe(2);
    expect(result.error).toBeUndefined();
  });

  it("returns nodeCount=3 when tp * pp = 12 (three nodes)", () => {
    const result = resolveGpuCount(undefined, { tensorParallelSize: 4, pipelineParallelSize: 3 });
    expect(result.gpuCount).toBe(12);
    expect(result.nodeCount).toBe(3);
  });

  it("CLI --gpus determines nodeCount when overriding YAML", () => {
    const result = resolveGpuCount(4, { tensorParallelSize: 4, pipelineParallelSize: 2 });
    expect(result.gpuCount).toBe(4);
    expect(result.nodeCount).toBe(1);
    expect(result.error).toBeUndefined();
  });

  it("respects a custom gpusPerNode when computing nodeCount", () => {
    const result = resolveGpuCount(undefined, { tensorParallelSize: 8 }, 8);
    expect(result.error).toBeUndefined();
    expect(result.gpuCount).toBe(8);
    expect(result.nodeCount).toBe(1);
  });
});

describe("stripIvllmKeys", () => {
  it("removes min-vllm-version from the output YAML", () => {
    const path = writeTmp("model: some/model\nmin-vllm-version: \"0.9.1\"\ntensor-parallel-size: 4\n");
    try {
      const result = stripIvllmKeys(path);
      expect(result).not.toContain("min-vllm-version");
    } finally { unlinkSync(path); }
  });

  it("preserves all non-ivllm keys", () => {
    const path = writeTmp(
      "model: Qwen/Qwen2.5-0.5B-Instruct\n" +
      "min-vllm-version: \"0.9.1\"\n" +
      "tensor-parallel-size: 2\n" +
      "max-model-len: 32768\n"
    );
    try {
      const result = stripIvllmKeys(path);
      expect(result).toContain("Qwen/Qwen2.5-0.5B-Instruct");
      expect(result).toContain("tensor-parallel-size");
      expect(result).toContain("max-model-len");
    } finally { unlinkSync(path); }
  });

  it("is a no-op when no ivllm-only keys are present", () => {
    const path = writeTmp("model: some/model\ntensor-parallel-size: 4\n");
    try {
      const result = stripIvllmKeys(path);
      expect(result).toContain("some/model");
      expect(result).toContain("tensor-parallel-size");
    } finally { unlinkSync(path); }
  });

  it("IVLLM_ONLY_KEYS contains min-vllm-version", () => {
    expect(IVLLM_ONLY_KEYS.has("min-vllm-version")).toBe(true);
  });
});

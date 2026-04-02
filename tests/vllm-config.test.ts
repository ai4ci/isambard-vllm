import { describe, it, expect } from "bun:test";
import { writeFileSync, unlinkSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { parseVllmConfig } from "../src/vllm-config.ts";

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
});

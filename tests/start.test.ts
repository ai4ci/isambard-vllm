import { describe, it, expect } from "bun:test";
import { parseJobDetails, hfCachePath, parseStartArgs } from "../src/job.ts";

describe("parseJobDetails", () => {
  it("parses a complete running job", () => {
    const json = JSON.stringify({
      status: "running",
      job_name: "my-job",
      slurm_job_id: "12345",
      compute_hostname: "compute01",
      model: "Qwen/Qwen2.5-0.5B-Instruct",
      server_port: 8000,
    });
    const result = parseJobDetails(json);
    expect(result?.status).toBe("running");
    expect(result?.slurm_job_id).toBe("12345");
    expect(result?.compute_hostname).toBe("compute01");
    expect(result?.server_port).toBe(8000);
  });

  it("parses a pending job with only required fields", () => {
    const json = JSON.stringify({ status: "pending", job_name: "my-job" });
    const result = parseJobDetails(json);
    expect(result?.status).toBe("pending");
    expect(result?.job_name).toBe("my-job");
  });

  it("parses a failed job with error field", () => {
    const json = JSON.stringify({
      status: "failed",
      job_name: "my-job",
      error: "vLLM process died during startup",
    });
    const result = parseJobDetails(json);
    expect(result?.status).toBe("failed");
    expect(result?.error).toBe("vLLM process died during startup");
  });

  it("returns null for empty string", () => {
    expect(parseJobDetails("")).toBeNull();
  });

  it("returns null for invalid JSON", () => {
    expect(parseJobDetails("not json")).toBeNull();
  });

  it("returns null for JSON missing status field", () => {
    expect(parseJobDetails(JSON.stringify({ job_name: "x" }))).toBeNull();
  });
});

describe("hfCachePath", () => {
  it("builds path for org/model format", () => {
    expect(hfCachePath("/projects/p/hf", "Qwen/Qwen2.5-0.5B-Instruct"))
      .toBe("/projects/p/hf/hub/models--Qwen--Qwen2.5-0.5B-Instruct");
  });

  it("builds path for org with hyphens", () => {
    expect(hfCachePath("/projects/p/hf", "meta-llama/Llama-3-8b"))
      .toBe("/projects/p/hf/hub/models--meta-llama--Llama-3-8b");
  });

  it("builds path for model with no org", () => {
    expect(hfCachePath("/projects/p/hf", "gpt2"))
      .toBe("/projects/p/hf/hub/models--gpt2");
  });
});

describe("parseStartArgs", () => {
  it("parses required args (non-mock: --config only, model comes from YAML)", () => {
    const result = parseStartArgs(["my-job", "--config", "vllm.yaml"]);
    expect(result.jobName).toBe("my-job");
    expect(result.configFile).toBe("vllm.yaml");
    expect(result.model).toBeUndefined();
  });

  it("applies defaults for optional args", () => {
    const result = parseStartArgs(["my-job", "--config", "vllm.yaml"]);
    expect(result.gpuCount).toBeUndefined();
    expect(result.timeLimit).toBe("8:00:00");
    expect(result.serverPort).toBe(8000);
  });

  it("parses optional --local-port", () => {
    const result = parseStartArgs(["my-job", "--config", "c.yaml", "--local-port", "11435"]);
    expect(result.localPort).toBe(11435);
  });

  it("parses optional --gpus", () => {
    const result = parseStartArgs(["my-job", "--config", "c.yaml", "--gpus", "8"]);
    expect(result.gpuCount).toBe(8);
  });

  it("parses optional --time", () => {
    const result = parseStartArgs(["my-job", "--config", "c.yaml", "--time", "8:00:00"]);
    expect(result.timeLimit).toBe("8:00:00");
  });

  it("throws when job name is missing", () => {
    expect(() => parseStartArgs(["--config", "c.yaml"])).toThrow(/job name/i);
  });

  it("--config is optional in non-mock mode (resolved later from job store)", () => {
    const result = parseStartArgs(["my-job"]);
    expect(result.configFile).toBeUndefined();
    expect(result.mock).toBe(false);
  });

  it("--config is still accepted when provided", () => {
    const result = parseStartArgs(["my-job", "--config", "c.yaml"]);
    expect(result.configFile).toBe("c.yaml");
  });

  it("--dry-run flag sets dryRun: true", () => {
    const result = parseStartArgs(["my-job", "--config", "c.yaml", "--dry-run"]);
    expect(result.dryRun).toBe(true);
  });

  it("dryRun defaults to false when flag absent", () => {
    const result = parseStartArgs(["my-job", "--config", "c.yaml"]);
    expect(result.dryRun).toBe(false);
  });

  it("--mock flag sets mock: true", () => {
    const result = parseStartArgs(["my-job", "--model", "m", "--mock"]);
    expect(result.mock).toBe(true);
  });

  it("mock defaults to false when flag absent", () => {
    const result = parseStartArgs(["my-job", "--config", "c.yaml"]);
    expect(result.mock).toBe(false);
  });

  it("--mock requires --model", () => {
    expect(() => parseStartArgs(["my-job", "--mock"])).toThrow(/--model/);
  });

  it("--mock does not require --config", () => {
    expect(() => parseStartArgs(["my-job", "--model", "m", "--mock"])).not.toThrow();
  });

  it("--mock with --dry-run sets both flags", () => {
    const result = parseStartArgs(["my-job", "--model", "m", "--mock", "--dry-run"]);
    expect(result.mock).toBe(true);
    expect(result.dryRun).toBe(true);
  });

  it("--config is still accepted without --mock", () => {
    const result = parseStartArgs(["my-job", "--config", "c.yaml"]);
    expect(result.configFile).toBe("c.yaml");
  });

  it("--tensor-parallel-size is not accepted (use YAML config)", () => {
    // Should not error — unknown flags are silently ignored — but the field is not on StartArgs
    const result = parseStartArgs(["my-job", "--config", "c.yaml", "--tensor-parallel-size", "4"]);
    expect((result as Record<string, unknown>)["tensorParallelSize"]).toBeUndefined();
  });
});

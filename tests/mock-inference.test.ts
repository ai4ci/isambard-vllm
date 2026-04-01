import { describe, it, expect } from "bun:test";
import { renderMockInferenceScript } from "../src/templates/mock-inference.ts";

const base = {
  jobName: "mock-job",
  model: "Qwen/Qwen2.5-0.5B-Instruct",
  workDir: "/home/user/mock-job",
  serverPort: 8000,
  timeLimit: "1:00:00",
};

describe("renderMockInferenceScript", () => {
  it("sets SBATCH job name", () => {
    expect(renderMockInferenceScript(base)).toContain("#SBATCH --job-name=mock-job");
  });

  it("sets SBATCH time limit", () => {
    expect(renderMockInferenceScript(base)).toContain("#SBATCH --time=1:00:00");
  });

  it("redirects stdout/stderr to log file in workDir via exec", () => {
    expect(renderMockInferenceScript(base)).toContain('exec > "$WORK_DIR/mock-job.slurm.log" 2>&1');
  });

  it("does not request GPUs (mock needs no GPU)", () => {
    expect(renderMockInferenceScript(base)).not.toContain("--gpus");
  });

  it("does not load CUDA module (mock needs no GPU)", () => {
    expect(renderMockInferenceScript(base)).not.toContain("module load");
  });

  it("does not activate a venv (mock has no venv dependency)", () => {
    expect(renderMockInferenceScript(base)).not.toContain("activate");
  });

  it("writes initialising status to job_details.json", () => {
    expect(renderMockInferenceScript(base)).toContain('"initialising"');
  });

  it("writes SLURM job ID to job_details.json", () => {
    expect(renderMockInferenceScript(base)).toContain("SLURM_JOB_ID");
  });

  it("writes compute hostname to job_details.json", () => {
    expect(renderMockInferenceScript(base)).toContain("compute_hostname");
  });

  it("updates status to running after startup delay", () => {
    expect(renderMockInferenceScript(base)).toContain('"running"');
  });

  it("starts HTTP server on correct port", () => {
    expect(renderMockInferenceScript(base)).toContain("8000");
  });

  it("serves /health endpoint", () => {
    expect(renderMockInferenceScript(base)).toContain("/health");
  });

  it("serves /v1/models endpoint", () => {
    expect(renderMockInferenceScript(base)).toContain("/v1/models");
  });

  it("includes the model name in the mock response", () => {
    expect(renderMockInferenceScript(base)).toContain("Qwen/Qwen2.5-0.5B-Instruct");
  });

  it("simulates startup delay before marking running", () => {
    const script = renderMockInferenceScript(base);
    // Default delay should appear as a sleep command
    expect(script).toMatch(/sleep\s+5/);
  });

  it("respects a custom startupDelaySecs", () => {
    const script = renderMockInferenceScript({ ...base, startupDelaySecs: 30 });
    expect(script).toMatch(/sleep\s+30/);
  });

  it("default startup delay is 5 seconds", () => {
    const script = renderMockInferenceScript(base);
    // default is 5
    expect(script).toMatch(/sleep\s+5/);
  });

  it("does not contain SSH tunnel logic", () => {
    const script = renderMockInferenceScript(base);
    expect(script).not.toContain("ssh -");
    expect(script).not.toContain("-R ");
  });

  it("respects a different server port", () => {
    const script = renderMockInferenceScript({ ...base, serverPort: 9000 });
    expect(script).toContain("9000");
    expect(script).not.toContain("8000");
  });
});

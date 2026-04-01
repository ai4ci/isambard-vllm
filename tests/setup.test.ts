import { describe, it, expect } from "bun:test";
import { renderSetupScript } from "../src/templates/setup.ts";
import { parseJobId, parseJobState } from "../src/slurm.ts";

describe("renderSetupScript", () => {
  const base = {
    venvPath: "/home/user/ivllm-venv/.venv",
    vllmVersion: "0.15.1",
    outputFile: "/home/user/ivllm-setup.log",
  };

  it("renders venv parent directory from venvPath", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("VENV_PARENT=/home/user/ivllm-venv");
  });

  it("renders venv path", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("VENV_PATH=/home/user/ivllm-venv/.venv");
  });

  it("renders vllm version", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("vllm[flashinfer]==0.15.1");
  });

  it("renders output file in SBATCH directive", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("#SBATCH --output=/home/user/ivllm-setup.log");
  });

  it("includes IVLLM_SETUP_SUCCESS marker", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("IVLLM_SETUP_SUCCESS");
  });

  it("does not use --pty flag (not valid in batch jobs)", () => {
    const script = renderSetupScript(base);
    expect(script).not.toContain("--pty");
  });

  it("activates the venv before installing", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("source /home/user/ivllm-venv/.venv/bin/activate");
  });
});

describe("parseJobId", () => {
  it("parses job ID from sbatch success output", () => {
    expect(parseJobId("Submitted batch job 12345\n")).toBe("12345");
  });

  it("parses job ID with no trailing newline", () => {
    expect(parseJobId("Submitted batch job 99999")).toBe("99999");
  });

  it("returns null for unrecognised output", () => {
    expect(parseJobId("sbatch: error: ...")).toBeNull();
  });
});

describe("parseJobState", () => {
  it("returns completed for COMPLETED state", () => {
    expect(parseJobState("COMPLETED")).toBe("completed");
  });

  it("returns running for RUNNING state", () => {
    expect(parseJobState("RUNNING")).toBe("running");
  });

  it("returns running for PENDING state", () => {
    expect(parseJobState("PENDING")).toBe("running");
  });

  it("returns failed for FAILED state", () => {
    expect(parseJobState("FAILED")).toBe("failed");
  });

  it("returns failed for TIMEOUT state", () => {
    expect(parseJobState("TIMEOUT")).toBe("failed");
  });

  it("returns failed for CANCELLED state", () => {
    expect(parseJobState("CANCELLED by 1000")).toBe("failed");
  });

  it("returns null for empty/unknown output", () => {
    expect(parseJobState("")).toBeNull();
  });
});

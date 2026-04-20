import { describe, it, expect } from "bun:test";
import { renderSetupScript } from "../src/templates/setup.ts";
import { parseJobId, parseJobState } from "../src/slurm.ts";

describe("renderSetupScript", () => {
  const base = { vllmVersion: "0.19.1" };

  it("is a CPU-only SLURM job (no --gpus directive)", () => {
    const script = renderSetupScript(base);
    expect(script).not.toContain("#SBATCH --gpus");
  });

  it("installs HPC SDK to $PROJECTDIR/ivllm/nvhpc", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("$PROJECTDIR/ivllm/nvhpc");
    expect(script).toContain("nvhpc_2026_263_Linux_aarch64_cuda_13.1");
  });

  it("skips HPC SDK install if $PROJECTDIR/ivllm/nvhpc already exists", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("$PROJECTDIR/ivllm/nvhpc");
    // idempotency guard
    expect(script).toMatch(/if \[ ! -d.*nvhpc/);
  });

  it("sets NVHPC_ROOT and LD_LIBRARY_PATH with compat path first", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3");
    expect(script).toContain("$NVHPC_ROOT/cuda/13.1/compat");
    // compat must appear before lib64 in LD_LIBRARY_PATH
    const idx1 = script.indexOf("cuda/13.1/compat");
    const idx2 = script.indexOf("cuda/13.1/lib64");
    expect(idx1).toBeLessThan(idx2);
  });

  it("loads gcc-native/14.2 module before pip install", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("module load gcc-native/14.2");
  });

  it("creates versioned venv at $PROJECTDIR/ivllm/<version>", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("$PROJECTDIR/ivllm/0.19.1");
  });

  it("skips venv install if versioned dir already exists", () => {
    const script = renderSetupScript(base);
    expect(script).toMatch(/if \[ ! -d.*\$PROJECTDIR\/ivllm\/0\.19\.1/);
  });

  it("installs vllm using cu130 wheels", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("wheels.vllm.ai/cu130");
    expect(script).not.toContain("cu129");
  });

  it("installs exact vllm version", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("vllm==0.19.1");
  });

  it("does not reference singularity", () => {
    const script = renderSetupScript(base);
    expect(script).not.toContain("singularity");
  });

  it("redirects output to log file via exec", () => {
    const script = renderSetupScript(base);
    expect(script).toContain('exec > "$HOME/.config/ivllm/setup.log" 2>&1');
    expect(script).not.toContain("#SBATCH --output");
  });

  it("includes IVLLM_SETUP_SUCCESS marker", () => {
    const script = renderSetupScript(base);
    expect(script).toContain("IVLLM_SETUP_SUCCESS");
  });

  it("does not use --pty flag", () => {
    const script = renderSetupScript(base);
    expect(script).not.toContain("--pty");
  });

  it("uses a different version when specified", () => {
    const script = renderSetupScript({ vllmVersion: "0.10.0" });
    expect(script).toContain("$PROJECTDIR/ivllm/0.10.0");
    expect(script).toContain("vllm==0.10.0");
    expect(script).not.toContain("0.19.1");
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

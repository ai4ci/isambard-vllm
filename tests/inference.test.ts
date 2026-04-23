import { describe, it, expect } from "bun:test";
import { renderInferenceScript } from "../src/templates/inference.ts";

const base = {
  jobName: "my-job",
  model: "Qwen/Qwen2.5-0.5B-Instruct",
  vllmVersion: "0.19.1",
  hfHome: "/projects/myproject/hf",
  configFileName: "vllm.yaml",
  workDir: "/home/user/my-job",
  serverPort: 8000,
  gpuCount: 4,
  nodeCount: 1,
  timeLimit: "4:00:00",
};

describe("renderInferenceScript", () => {
  it("sets SBATCH job name", () => {
    expect(renderInferenceScript(base)).toContain("#SBATCH --job-name=my-job");
  });

  it("sets SBATCH GPU count", () => {
    expect(renderInferenceScript(base)).toContain("#SBATCH --gpus=4");
  });

  it("sets SBATCH time limit", () => {
    expect(renderInferenceScript(base)).toContain("#SBATCH --time=4:00:00");
  });

  it("redirects stdout/stderr to log file in workDir via exec", () => {
    expect(renderInferenceScript(base)).toContain('exec > "/home/user/my-job/my-job.slurm.log" 2>&1');
  });

  it("activates the versioned venv from $PROJECTDIR", () => {
    expect(renderInferenceScript(base)).toContain(
      "source $PROJECTDIR/ivllm/0.19.1/bin/activate"
    );
  });

  it("sets NVHPC_ROOT before venv activation", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain("NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3");
    const idxNvhpc = script.indexOf("NVHPC_ROOT=");
    const idxActivate = script.indexOf("source $PROJECTDIR/ivllm/");
    expect(idxNvhpc).toBeLessThan(idxActivate);
  });

  it("sets CUDA_HOME and adds nvcc to PATH for Ray worker kernel compilation", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain("CUDA_HOME=$NVHPC_ROOT/cuda/12.9");
    expect(script).toContain("PATH=$CUDA_HOME/bin:$PATH");
  });

  it("sets CPATH to include NVHPC math_libs headers for cublasLt.h", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain("CPATH=$NVHPC_ROOT/math_libs/12.9/include:");
  });

  it("loads gcc-native module for C++20 host compiler support", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain("module load brics/nccl gcc-native");
  });

  it("sets LD_LIBRARY_PATH with cuda/12.9/compat first", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain("$NVHPC_ROOT/cuda/12.9/compat");
    const idxCompat = script.indexOf("cuda/12.9/compat");
    const idxLib64 = script.indexOf("cuda/12.9/lib64");
    expect(idxCompat).toBeLessThan(idxLib64);
  });

  it("does not reference singularity", () => {
    expect(renderInferenceScript(base)).not.toContain("singularity");
  });

  it("does not reference cu130", () => {
    expect(renderInferenceScript(base)).not.toContain("cu130");
  });

  it("sets HF_HOME", () => {
    expect(renderInferenceScript(base)).toContain(
      "export HF_HOME=/projects/myproject/hf"
    );
  });

  it("serves the correct model", () => {
    expect(renderInferenceScript(base)).toContain("Qwen/Qwen2.5-0.5B-Instruct");
  });

  it("uses the correct server port", () => {
    expect(renderInferenceScript(base)).toContain("--port 8000");
  });

  it("uses vllm serve --config (no model positional arg on command line)", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain("vllm serve");
    expect(script).toContain('--config "$VLLM_CONFIG"');
    // model positional on the vllm CLI would conflict with YAML; not present
    expect(script).not.toContain("vllm serve Qwen");
  });

  it("does not pass --tensor-parallel-size on command line (comes from YAML config)", () => {
    expect(renderInferenceScript(base)).not.toContain("--tensor-parallel-size");
  });

  it("does not pass --served-model-name on command line", () => {
    expect(renderInferenceScript(base)).not.toContain("--served-model-name");
  });

  it("references the vllm config file from workDir", () => {
    expect(renderInferenceScript(base)).toContain("/home/user/my-job/vllm.yaml");
  });

  it("writes initialising status to job_details.json", () => {
    expect(renderInferenceScript(base)).toContain('"initialising"');
  });

  it("writes compute hostname to job_details.json", () => {
    expect(renderInferenceScript(base)).toContain("compute_hostname");
  });

  it("writes SLURM job ID to job_details.json", () => {
    expect(renderInferenceScript(base)).toContain("SLURM_JOB_ID");
  });

  it("updates status to running after health check passes", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('"running"');
  });

  it("updates status to failed when vLLM process dies during startup", () => {
    expect(renderInferenceScript(base)).toContain('"failed"');
  });

  it("updates status to timeout when health check times out", () => {
    expect(renderInferenceScript(base)).toContain('"timeout"');
  });

  it("polls /health endpoint on localhost", () => {
    expect(renderInferenceScript(base)).toContain("localhost:8000/health");
  });

  it("does not contain SSH tunnel logic", () => {
    const script = renderInferenceScript(base);
    expect(script).not.toContain("ssh -");
    expect(script).not.toContain("-R ");
  });

  it("does not use --pty flag", () => {
    expect(renderInferenceScript(base)).not.toContain("--pty");
  });

  it("respects a different server port", () => {
    const script = renderInferenceScript({ ...base, serverPort: 9000 });
    expect(script).toContain("--port 9000");
    expect(script).toContain("localhost:9000/health");
  });

  it("respects a different gpu count in SBATCH directive", () => {
    const script = renderInferenceScript({ ...base, gpuCount: 8 });
    expect(script).toContain("#SBATCH --gpus=8");
  });
});

const multiNodeBase = {
  ...base,
  gpuCount: 8,
  nodeCount: 2,
};

describe("renderInferenceScript (multi-node)", () => {
  it("sets --nodes=2 in SBATCH for 2-node job", () => {
    expect(renderInferenceScript(multiNodeBase)).toContain("#SBATCH --nodes=2");
  });

  it("sets total GPU count in SBATCH --gpus", () => {
    expect(renderInferenceScript(multiNodeBase)).toContain("#SBATCH --gpus=8");
  });

  it("starts Ray head node", () => {
    expect(renderInferenceScript(multiNodeBase)).toContain("ray start --block --head");
  });

  it("starts Ray worker nodes via srun", () => {
    expect(renderInferenceScript(multiNodeBase)).toContain("ray start --block --address");
  });

  it("wraps ray start commands in bash -c to guarantee venv PATH on compute nodes", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('bash -c "source');
    expect(script).not.toContain("env VLLM_HOST_IP");
  });

  it("sources the venv inside each bash -c ray start call", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('bash -c "source');
    // All bash -c invocations that call ray or vllm should source the venv
    const activateCount = (script.match(/bash -c "source[^"]*\/bin\/activate/g) ?? []).length;
    expect(activateCount).toBeGreaterThanOrEqual(3); // head, worker, vllm serve (plus ray status)
  });

  it("sets VLLM_HOST_IP inside bash -c for ray head", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain("VLLM_HOST_IP=$HEAD_NODE_IP ray start --block --head");
  });

  it("sets VLLM_HOST_IP inside bash -c for ray workers", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain("VLLM_HOST_IP=$WORKER_IP ray start --block --address");
  });

  it("runs vllm serve with --distributed-executor-backend ray", () => {
    expect(renderInferenceScript(multiNodeBase)).toContain("--distributed-executor-backend ray");
  });

  it("runs vllm serve via srun --overlap on the head node", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain("srun --overlap");
  });

  it("uses HEAD_NODE as the compute_hostname", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain("HEAD_NODE");
    expect(script).toContain("compute_hostname");
  });

  it("loads the brics/nccl and gcc-native modules", () => {
    expect(renderInferenceScript(multiNodeBase)).toContain("module load brics/nccl gcc-native");
  });

  it("sets NVHPC_ROOT and LD_LIBRARY_PATH preamble before ray start", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain("NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3");
    expect(script).toContain("$NVHPC_ROOT/cuda/12.9/compat");
    // preamble must appear before ray start
    const idxNvhpc = script.indexOf("NVHPC_ROOT=");
    const idxRay = script.indexOf("ray start");
    expect(idxNvhpc).toBeLessThan(idxRay);
  });

  it("does not set deprecated Ray env vars removed in vLLM 0.19.1", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).not.toContain("VLLM_USE_RAY_SPMD_WORKER");
    expect(script).not.toContain("VLLM_USE_RAY_COMPILED_DAG");
    expect(script).not.toContain("VLLM_USE_RAY_SPMD_HEAD");
  });

  it("single-node template is unchanged for nodeCount=1", () => {
    const script = renderInferenceScript(base);
    expect(script).not.toContain("ray start");
    expect(script).not.toContain("--distributed-executor-backend");
  });
});

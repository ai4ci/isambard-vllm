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

  it("requests full node memory in SBATCH", () => {
    expect(renderInferenceScript(base)).toContain("#SBATCH --mem=0");
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

  it("redirects FLASHINFER_JIT_CACHE_DIR to Lustre for reliable flock and persistent cache", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain("WORK_DIR=\"/home/user/my-job\"");
    expect(script).toContain("FLASHINFER_CACHE_ROOT=$WORK_DIR/ivllm/");
    expect(script).toContain("FLASHINFER_JIT_CACHE_DIR=$FLASHINFER_CACHE_ROOT/flashinfer_cache");
    const idxWorkDir = script.indexOf('WORK_DIR="/home/user/my-job"');
    const idxFlashinfer = script.indexOf("FLASHINFER_CACHE_ROOT=$WORK_DIR/ivllm/");
    expect(idxWorkDir).toBeLessThan(idxFlashinfer);
  });

  it("symlinks ~/.cache/flashinfer to Lustre so Ray actors inherit Lustre cache without env var", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain("ln -sfn $FLASHINFER_JIT_CACHE_DIR ~/.cache/flashinfer");
  });

  it("checks writable directories before launching vllm", () => {
    const script = renderInferenceScript(base);
    expect(script).not.toContain('assert_writable_dir "$PROJECTDIR/ivllm"');
    expect(script).toContain('assert_writable_dir "$FLASHINFER_CACHE_ROOT"');
    expect(script).toContain('assert_writable_dir "$FLASHINFER_JIT_CACHE_DIR"');
    expect(script).toContain('assert_writable_dir "$HOME/.cache"');
    expect(script).toContain('assert_writable_dir "$WORK_DIR"');
    expect(script).toContain('assert_writable_dir "$HF_HOME"');
  });

  it("marks job failed before exiting on early writable-directory errors", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('fail_job "Required directory is not writable: $dir"');
    expect(script).toContain('jq --arg error "$error"');
    expect(script).toContain(".status = \"failed\"");
  });

  it("sets CC=gcc and CXX=g++ for JIT compilation with gcc-native module", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain("export CC=gcc");
    expect(script).toContain("export CXX=g++");
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

  it("sets HF_HUB_OFFLINE=1 to prevent API calls when model is already cached", () => {
    expect(renderInferenceScript(base)).toContain("export HF_HUB_OFFLINE=1");
  });

  it("symlinks shared plugins into the job work directory when present", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('if [ -d "$PROJECTDIR/ivllm/plugins" ]; then');
    expect(script).toContain('ln -sfn "$PROJECTDIR/ivllm/plugins" "$WORK_DIR/plugins"');
  });

  it("changes into the job work directory before starting vllm serve", () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('cd "$WORK_DIR"');
    const idxCd = script.indexOf('cd "$WORK_DIR"');
    const idxServe = script.indexOf("vllm serve");
    expect(idxCd).toBeLessThan(idxServe);
  });

  it("single-node: trap on_exit EXIT is not followed by prose text on the same line", () => {
    // Regression: the exit trap block was concatenated with a comment fragment,
    // causing bash to treat comment words as invalid signal names.
    const script = renderInferenceScript(base);
    const trapLine = script.split("\n").find(l => l.trimStart().startsWith("trap on_exit EXIT"));
    expect(trapLine).toBeDefined();
    expect(trapLine!.trim()).toBe("trap on_exit EXIT");
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

  it("waits indefinitely for health rather than enforcing a startup timeout", () => {
    const script = renderInferenceScript(base);
    expect(script).not.toContain("MAX_WAIT=");
    expect(script).not.toContain("Timed out waiting for vLLM");
    expect(script).not.toContain('"timeout"');
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

  it("requests GPUs per node in SBATCH for multi-node overlap compatibility", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain("#SBATCH --gpus-per-node=4");
    expect(script).not.toContain("#SBATCH --gpus=8");
  });

  it("starts Ray head node", () => {
    expect(renderInferenceScript(multiNodeBase)).toContain("ray start --block --head");
  });

  it("starts Ray worker nodes via srun", () => {
    expect(renderInferenceScript(multiNodeBase)).toContain("ray start --block --address");
  });

  it("requests full node memory in SBATCH", () => {
    expect(renderInferenceScript(multiNodeBase)).toContain("#SBATCH --mem=0");
  });

  it("requests full node memory for all multi-node srun steps", () => {
    const script = renderInferenceScript(multiNodeBase);
    const memRequests = script.match(/--mem=0/g) ?? [];
    expect(memRequests.length).toBeGreaterThanOrEqual(5); // SBATCH + head + worker + status + serve
  });

  it("caps Ray object store memory to reduce host-RAM pressure during startup", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain("RAY_OBJECT_STORE_MEMORY=$((64 * 1024 * 1024 * 1024))");
    expect(script).toContain("--object-store-memory=$RAY_OBJECT_STORE_MEMORY");
  });

  it("captures a slurm accounting snapshot in the job work directory on exit", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('WORK_DIR="/home/user/my-job"');
    expect(script).toContain('SLURM_ACCOUNTING_FILE="$WORK_DIR/slurm-accounting.txt"');
    expect(script).toContain('sacct -j "$SLURM_JOB_ID"');
  });

  it("archives per-node Ray logs from local scratch back to the job work directory on exit", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('RAY_LOG_ARCHIVE_DIR="$WORK_DIR/ray-logs"');
    expect(script).toContain('readlink -f /local/user/$UID/ray/session_latest');
    expect(script).toContain('cp -a "$RAY_SESSION_DIR/logs/." "$RAY_DESTINATION/"');
  });

  it("installs an EXIT trap so diagnostics are still collected after startup failures", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain("on_exit()");
    expect(script).toContain("trap on_exit EXIT");
  });

  it("records per-node archive status files even when Ray log collection fails", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('ARCHIVE_STATUS_FILE="$RAY_DESTINATION/archive-status.txt"');
    expect(script).toContain('printf "%s\\n" "Starting Ray log archival for $NODE_NAME"');
    expect(script).toContain('printf "%s\\n" "Ray log archival srun failed for $NODE_NAME"');
  });

  it("collects exit diagnostics explicitly before the scripted startup-failure exit", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('finalize_and_exit 1 "startup failure"');
    expect(script).toContain('collect_exit_diagnostics()');
  });

  it("symlinks shared plugins into the multi-node job work directory when present", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('ln -sfn "$PROJECTDIR/ivllm/plugins" "$WORK_DIR/plugins"');
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

  it("changes into the job work directory before multi-node vllm serve", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain(`bash -c "cd ${multiNodeBase.workDir}`);
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

  it("sets NCCL_CROSS_NIC=1 and NCCL_FORCE_FLUSH=0 for multi-node NCCL comms", () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain("NCCL_CROSS_NIC=1");
    expect(script).toContain("NCCL_FORCE_FLUSH=0");
  });

  it("single-node template is unchanged for nodeCount=1", () => {
    const script = renderInferenceScript(base);
    expect(script).not.toContain("ray start");
    expect(script).not.toContain("--distributed-executor-backend");
  });
});

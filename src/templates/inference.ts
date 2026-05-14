import { SACCT_DIAGNOSTICS_FORMAT } from "../slurm.ts";

export interface InferenceScriptOptions {
  jobName: string;
  model: string;
  vllmVersion: string;
  hfHome: string;
  configFileName: string;
  workDir: string;
  serverPort: number;
  gpuCount: number;
  nodeCount: number;
  timeLimit: string;
}

const NVHPC_PREAMBLE = `export NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3
export CUDA_HOME=$NVHPC_ROOT/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
# NVHPC separates math library headers (cuBLAS, cuSPARSE) from the CUDA SDK headers.
# flashinfer JIT kernels include cublasLt.h which is in math_libs, not cuda/include.
export CPATH=$NVHPC_ROOT/math_libs/12.9/include:\${CPATH:-}
export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64:\${LD_LIBRARY_PATH:-}
# Use gcc from gcc-native module for JIT compilation (flashinfer, torch.compile).
export CC=gcc
export CXX=g++
# Redirect flashinfer JIT cache to Lustre (PROJECTDIR) instead of NFS home (~/.cache).
# NFS does not support fcntl.flock reliably; Lustre does. Cache persists across jobs.
export FLASHINFER_JIT_CACHE_DIR=$LOCALDIR/ivllm/flashinfer_cache
# Symlink ~/.cache/flashinfer -> Lustre so that Ray actors (which don't inherit
# FLASHINFER_JIT_CACHE_DIR from vLLM's ray_env.py propagation list) also use Lustre.
mkdir -p $PROJECTDIR/ivllm/flashinfer_cache ~/.cache
if [ -d ~/.cache/flashinfer ] && [ ! -L ~/.cache/flashinfer ]; then
  cp -r ~/.cache/flashinfer/. $PROJECTDIR/ivllm/flashinfer_cache/ 2>/dev/null || true
  rm -rf ~/.cache/flashinfer
fi
ln -sfn $PROJECTDIR/ivllm/flashinfer_cache ~/.cache/flashinfer`;

function renderExitDiagnostics(_workDir: string): string {
  return `SLURM_ACCOUNTING_FILE="$WORK_DIR/slurm-accounting.txt"

persist_slurm_accounting() {
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$SLURM_JOB_ID" --format=${SACCT_DIAGNOSTICS_FORMAT} > "$SLURM_ACCOUNTING_FILE" 2>&1 || true
  fi
}`;
}

function renderWorkDirSetup(workDir: string): string {
  return `WORK_DIR="${workDir}"
mkdir -p "$WORK_DIR"
if [ -d "$PROJECTDIR/ivllm/plugins" ]; then
  ln -sfn "$PROJECTDIR/ivllm/plugins" "$WORK_DIR/plugins"
fi`;
}

function renderMultiNodeExitDiagnostics(workDir: string): string {
  return `${renderExitDiagnostics(workDir)}
RAY_LOG_ARCHIVE_DIR="$WORK_DIR/ray-logs"

persist_ray_logs() {
  mkdir -p "$RAY_LOG_ARCHIVE_DIR"
  for NODE_NAME in $(scontrol show hostnames $SLURM_NODELIST); do
    RAY_DESTINATION="$RAY_LOG_ARCHIVE_DIR/$NODE_NAME"
    ARCHIVE_STATUS_FILE="$RAY_DESTINATION/archive-status.txt"
    mkdir -p "$RAY_DESTINATION"
    printf "%s\\n" "Starting Ray log archival for $NODE_NAME" > "$ARCHIVE_STATUS_FILE"
    if srun --overlap \\
      --nodelist "$NODE_NAME" \\
      --nodes=1 \\
      --ntasks-per-node 1 \\
      --cpus-per-task 1 \\
      --export=ALL,RAY_DESTINATION="$RAY_DESTINATION" \\
      bash -lc '
        RAY_SESSION_DIR=$(readlink -f /local/user/$UID/ray/session_latest 2>/dev/null || true)
        mkdir -p "$RAY_DESTINATION"
        if [ -n "$RAY_SESSION_DIR" ] && [ -d "$RAY_SESSION_DIR/logs" ]; then
          cp -a "$RAY_SESSION_DIR/logs/." "$RAY_DESTINATION/" 2>/dev/null || true
          printf "%s\\n" "$RAY_SESSION_DIR" > "$RAY_DESTINATION/session_dir.txt"
        else
          printf "%s\\n" "No Ray logs found at /local/user/$UID/ray/session_latest" > "$RAY_DESTINATION/missing.txt"
        fi
      '; then
      printf "%s\\n" "Finished Ray log archival for $NODE_NAME" >> "$ARCHIVE_STATUS_FILE"
    else
      printf "%s\\n" "Ray log archival srun failed for $NODE_NAME" >> "$ARCHIVE_STATUS_FILE"
    fi
  done
}`;
}

function renderExitTrap(includeRayLogs: boolean): string {
  const maybePersistRayLogs = includeRayLogs ? "\n  persist_ray_logs" : "";
  return `collect_exit_diagnostics() {
  local reason="$1"
  echo "Collecting exit diagnostics ($reason)..."
  persist_slurm_accounting${maybePersistRayLogs}
  echo "Finished exit diagnostics ($reason)."
}

finalize_and_exit() {
  local exit_code="$1"
  local reason="$2"
  trap - EXIT
  collect_exit_diagnostics "$reason"
  exit "$exit_code"
}

on_exit() {
  EXIT_CODE=$?
  trap - EXIT
  collect_exit_diagnostics "EXIT trap"
  exit $EXIT_CODE
}

trap on_exit EXIT`;
}

function renderHealthCheckAndWait(workDir: string, serverPort: number): string {
  return `# Poll /health until vLLM is ready
echo "Waiting for vLLM to become healthy on port ${serverPort}..."
while true; do
  if curl -sf http://localhost:${serverPort}/health > /dev/null 2>&1; then
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM process died during startup"
    jq '.status = "failed" | .error = "vLLM process died during startup"' \\
      "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
    finalize_and_exit 1 "startup failure"
  fi
  sleep 1
done

echo "vLLM is ready"
jq '.status = "running"' "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"

# Keep SLURM job alive while vLLM runs
wait $VLLM_PID
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  jq --arg error "vLLM exited with code $EXIT_CODE" \\
    '.status = "failed" | .error = $error' \\
    "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
fi

finalize_and_exit $EXIT_CODE "vLLM exit"`;
}

function renderSingleNodeScript(opts: InferenceScriptOptions): string {
  const { jobName, model, vllmVersion, hfHome, configFileName, workDir, serverPort, gpuCount, timeLimit } = opts;
  const venvPath = `$PROJECTDIR/ivllm/${vllmVersion}`;

  return `#!/bin/bash
#SBATCH --job-name=${jobName}
#SBATCH --nodes=1
#SBATCH --gpus=${gpuCount}
#SBATCH --mem=0
#SBATCH --time=${timeLimit}
#SBATCH --exclusive

exec > "${workDir}/${jobName}.slurm.log" 2>&1

JOB_DETAILS="${workDir}/job_details.json"
VLLM_CONFIG="${workDir}/${configFileName}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
SERVER_PORT=${serverPort}
COMPUTE_HOSTNAME=$(hostname)

# Write initialising status — LOCAL already created the file with "pending";
# we overwrite with full details now that we have the SLURM context.
jq -n \\
  --arg status "initialising" \\
  --arg job_name "${jobName}" \\
  --arg slurm_job_id "$SLURM_JOB_ID" \\
  --arg compute_hostname "$COMPUTE_HOSTNAME" \\
  --arg model "${model}" \\
  --argjson server_port ${serverPort} \\
  '{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
    compute_hostname: $compute_hostname, model: $model, server_port: $server_port}' \\
  > "$JOB_DETAILS"

module load brics/nccl gcc-native

${NVHPC_PREAMBLE}
source ${venvPath}/bin/activate
export HF_HOME=${hfHome}
export HF_HUB_OFFLINE=1
${renderWorkDirSetup(workDir)}
${renderExitDiagnostics(workDir)}
${renderExitTrap(false)}
cd "$WORK_DIR"

# vLLM is launched via srun in the background — model, tensor-parallel-size, and all tuning
# options come from the config file; host and port are infrastructure overrides.
srun \\
  --nodes=1 \\
  --gpus=${gpuCount} \\
  --mem=0 \\
  --cpus-per-task 72 \\
  --ntasks-per-node 1 \\
  vllm serve \\
  --config "$VLLM_CONFIG" \\
  --host 0.0.0.0 \\
  --port ${serverPort} \\
  &
VLLM_PID=$!

${renderHealthCheckAndWait(workDir, serverPort)}`;
}

function renderMultiNodeScript(opts: InferenceScriptOptions): string {
  const { jobName, model, vllmVersion, hfHome, configFileName, workDir, serverPort, gpuCount, nodeCount, timeLimit } = opts;
  const gpusPerNode = Math.floor(gpuCount / nodeCount);
  const venvPath = `$PROJECTDIR/ivllm/${vllmVersion}`;
  const rayObjectStoreMemoryGiB = 64;

  return `#!/bin/bash
#SBATCH --job-name=${jobName}
#SBATCH --nodes=${nodeCount}
#SBATCH --gpus-per-node=${gpusPerNode}
#SBATCH --mem=0
#SBATCH --time=${timeLimit}
#SBATCH --exclusive

exec > "${workDir}/${jobName}.slurm.log" 2>&1

JOB_DETAILS="${workDir}/job_details.json"
VLLM_CONFIG="${workDir}/${configFileName}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
SERVER_PORT=${serverPort}
GPUS_PER_NODE=${gpusPerNode}
HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
COMPUTE_HOSTNAME=$HEAD_NODE
HEAD_NODE_IP=$(dig +short $HEAD_NODE)
RAY_PORT=6378

# Write initialising status — LOCAL already created the file with "pending";
# we overwrite with full details now that we have the SLURM context.
jq -n \\
  --arg status "initialising" \\
  --arg job_name "${jobName}" \\
  --arg slurm_job_id "$SLURM_JOB_ID" \\
  --arg compute_hostname "$COMPUTE_HOSTNAME" \\
  --arg model "${model}" \\
  --argjson server_port ${serverPort} \\
  '{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
    compute_hostname: $compute_hostname, model: $model, server_port: $server_port}' \\
  > "$JOB_DETAILS"

module load brics/nccl gcc-native

${NVHPC_PREAMBLE}
source ${venvPath}/bin/activate
export HF_HOME=${hfHome}
export HF_HUB_OFFLINE=1
${renderWorkDirSetup(workDir)}
${renderMultiNodeExitDiagnostics(workDir)}
${renderExitTrap(true)}
RAY_OBJECT_STORE_MEMORY=$((${rayObjectStoreMemoryGiB} * 1024 * 1024 * 1024))

# Required env vars for multi-node Ray+vLLM
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_CROSS_NIC=1
export NCCL_FORCE_FLUSH=0

# Start Ray head node
# bash -c is used to guarantee venv PATH is active on the compute node,
# and to avoid any .local/bin/env shadowing /usr/bin/env on login nodes.
echo "Starting Ray head node ($HEAD_NODE)..."
srun \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --mem=0 \\
  --cpus-per-task 72 \\
  --ntasks-per-node 1 \\
  bash -c "source ${venvPath}/bin/activate && VLLM_HOST_IP=$HEAD_NODE_IP ray start --block --head --node-ip-address=$HEAD_NODE_IP --port=$RAY_PORT --object-store-memory=$RAY_OBJECT_STORE_MEMORY" &
sleep 20

# Start Ray worker nodes
WORKER_NODES=$(scontrol show hostnames $SLURM_NODELIST | tail -n+2)
for WORKER in $WORKER_NODES; do
  WORKER_IP=$(dig +short $WORKER)
  echo "Starting Ray worker node: $WORKER ($WORKER_IP)"
  srun \\
    --nodelist "$WORKER" \\
    --nodes=1 \\
    --gpus=$GPUS_PER_NODE \\
    --mem=0 \\
    --cpus-per-task 72 \\
    --ntasks-per-node 1 \\
    bash -c "source ${venvPath}/bin/activate && VLLM_HOST_IP=$WORKER_IP ray start --block --address=$HEAD_NODE_IP:$RAY_PORT --node-ip-address=$WORKER_IP --object-store-memory=$RAY_OBJECT_STORE_MEMORY" &
done
sleep 20

# Verify Ray cluster is ready
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --mem=0 \\
  --ntasks-per-node 1 \\
  bash -c "source ${venvPath}/bin/activate && ray status"

# Start vLLM on the head node via srun --overlap (runs within existing job allocation)
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --mem=0 \\
  --ntasks-per-node 1 \\
  bash -c "cd ${workDir} && source ${venvPath}/bin/activate && VLLM_HOST_IP=$HEAD_NODE_IP vllm serve --config ${workDir}/${configFileName} --distributed-executor-backend ray --host 0.0.0.0 --port ${serverPort}" \\
  &
VLLM_PID=$!

${renderHealthCheckAndWait(workDir, serverPort)}`;
}

export function renderInferenceScript(opts: InferenceScriptOptions): string {
  return (opts.nodeCount > 1 ? renderMultiNodeScript(opts) : renderSingleNodeScript(opts)).trimStart();
}

import { SACCT_DIAGNOSTICS_FORMAT } from '../slurm.ts';
import type { EnvVarEntry, SessionState, Paths } from '../../src/types.ts';

const NVHPC_PREAMBLE = `
export NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3
export CUDA_HOME=$NVHPC_ROOT/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
# NVHPC separates math library headers (cuBLAS, cuSPARSE) from the CUDA SDK headers.
# flashinfer JIT kernels include cublasLt.h which is in math_libs, not cuda/include.
export CPATH=$NVHPC_ROOT/math_libs/12.9/include:\${CPATH:-}
export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64:\${LD_LIBRARY_PATH:-}
# Use gcc from gcc-native module for JIT compilation (flashinfer, torch.compile).
export CC=gcc
export CXX=g++
# Prevent torch from over-subscribing CPU cores across parallel workers.
# GH200 has 72 cores; 16 threads/worker is safe for the 4-GPU-per-node case.
export OMP_NUM_THREADS=16
`;

/**
 * Renders user-supplied env vars as `export KEY="VALUE"` lines.
 * @param envVars
 */
function renderEnvVars(envVars: EnvVarEntry[]): string {
  if (envVars.length === 0) return '';
  const lines = envVars.map((e) => `export ${e.key}="${e.value}"`);
  return `# User-supplied environment variables\n${lines.join('\n')}`;
}

/**
 *
 */
function renderExitDiagnostics(): string {
  return `SLURM_ACCOUNTING_FILE="$WORK_DIR/slurm-accounting.txt"

persist_slurm_accounting() {
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$SLURM_JOB_ID" --format=${SACCT_DIAGNOSTICS_FORMAT} > "$SLURM_ACCOUNTING_FILE" 2>&1 || true
  fi
}`;
}

// function renderRelocateCache(
//   env: string,
//   cache: string,
//   defaultLocation: string,
//   model: string,
// ) {
//   const modelDir = model.replaceAll('/', '_').replaceAll('.', '_');
//   // N.B. renderArchBase must be called before this is.
//   return `export ${env}=$SCRATCHDIR/${modelDir}_$ARCH_SUFFIX/${cache}
// # Symlink ${defaultLocation} -> Lustre so that Ray actors (which don't inherit ${env} from vLLM's ray_env.py propagation list) also use it.
// mkdir -p "$${env}" "$(dirname ${defaultLocation})"
// ln -sfn "$${env}" ${defaultLocation}`;
// }

/**
 *
 * @param workDir
 * @param model
 */
function renderWorkDirSetup(paths: Paths): string {
  return `
WORK_DIR="${paths.remoteJobDir}"
mkdir -p "${paths.remoteJobDir}"
# Look for and link the plugins directory
if [ -d "${paths.remoteProjectVllmPluginsDir}" ]; then
  ln -sfn "${paths.remoteProjectVllmPluginsDir}" "${paths.remoteJobVllmPluginsDir}"
fi

# 1. Compute cache key from vllm.yaml
TAR_FILE="${paths.remoteProjectJobCacheFile}"

# 2. Move $HOME to fast local tmpfs
export HOME="$LOCALDIR/user_home"
mkdir -p "$HOME"

# 3. Try to restore cached JIT compilations
if [ -f "$TAR_FILE" ]; then
  echo "Restoring JIT cache from shared storage..."
  tar xzf "$TAR_FILE" -C "$LOCALDIR" 2>/dev/null && echo "Cache restored" \
  || echo "Cache corrupt — recompiling"
fi

export FLASHINFER_JIT_CACHE_DIR="$HOME/.cache/flashinfer"
export DG_JIT_CACHE_DIR="$HOME/.deep_gemm"
export TRITON_CACHE_DIR="$HOME/.triton"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"

export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
`;
}

// Called after renderWorkDirSetup so OK to use variables from that.
function renderMonitor(): string {
  const debug = !!process.env.IVLLM_DEBUG;
  if (debug) {
    return `
    export VLLM_LOGGING_LEVEL=DEBUG
    # to turn on more logging.
    export VLLM_LOG_STATS_INTERVAL=1.
    # to get log statistics more frequently for tracking running queue, waiting queue and cache hit states.

    export VLLM_TRACE_FUNCTION=1
    # to record all function calls for inspection in the log files to tell which function crashes or hangs. (WARNING: This flag will slow

    # 1. Force NVIDIA NCCL to print full initialization, connection, and topology maps
    export NCCL_DEBUG=TRACE
    export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
    export NCCL_DEBUG_FILE=$WORK_DIR/nccl_log.log

    # 2. Force PyTorch Distributed to log process group handshakes and tracking metrics
    export TORCH_CPP_LOG_LEVEL=INFO
    export TORCH_DISTRIBUTED_DEBUG=INFO
    export TORCH_SHOW_CPP_STACKTRACES=1

    # 3. Force Ray to log every internal actor IPC execution message
    export RAY_LOG_TO_DRIVER=1

    echo "=== Shared Memory Allocation ==="
    df -h /dev/shm
    # Start a background resource monitor
    (
      while true; do
        echo "--- Memory Snapshot at $(date) ---"
        # Filter by your Slurm Job ID, aggregate memory by process name, and sort
        ps -u "$USER" -o rss,comm | awk '
    NR>1 { mem[$2] += $1; count[$2]++ }
    END {
      for (cmd in mem) {
        printf "Cmd: %-15s | Count: %-2d | Total RAM: %.2f MB\\n", cmd, count[cmd], mem[cmd]/1024
      }
    }' | sort -k8 -nr | head -n 5

    # JIT Cache Growth Monitor
    echo "--- JIT Cache Sizes at $(date) ---"
    du -sh $FLASHINFER_JIT_CACHE_DIR $DG_JIT_CACHE_DIR $TRITON_CACHE_DIR $VLLM_CACHE_ROOT 2>/dev/null

    sleep 5
    done
    ) &
    MONITOR_PID=$!
    `;
  } else {
    // A terse memory and disk usage monitor for startup
    return `
    # Compact progress monitor
    (
      while true; do
        printf "[%s] RAM: %s | Cache: fi=%sK dg=%sK ti=%sK vc=%sK\n" \
        "$(date +%H:%M:%S)" \
        "$(ps -u $USER -o rss=,comm= | awk '{m[$2]+=$1} END{for(c in m) if(m[c]>1024) printf "%s=%dM ",c,m[c]/1024;
          else printf "%s=%dK ",c,m[c]}')" \
"$(du -sk $FLASHINFER_JIT_CACHE_DIR 2>/dev/null | cut -f1)" \
"$(du -sk $DG_JIT_CACHE_DIR 2>/dev/null | cut -f1)" \
"$(du -sk $TRITON_CACHE_DIR 2>/dev/null | cut -f1)" \
"$(du -sk $VLLM_CACHE_ROOT 2>/dev/null | cut -f1)"
sleep 20
done
    ) &
    MONITOR_PID=$!
    `;
  }
}

/**
 *
 * @param includeRayLogs
 */
function renderExitTrap(includeRayLogs: boolean): string {
  const maybePersistRayLogs = includeRayLogs ? '\n  persist_ray_logs' : '';
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

function renderWaitBlock(preCache: boolean) {
  return preCache
    ? `
# Keep SLURM job alive while vLLM runs
wait $VLLM_PID
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  jq --arg error "vLLM exited with code $EXIT_CODE" \\
  '.status = "failed" | .error = $error' \\
  "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
fi
`
    : '';
}

/**
 *
 * @param workDir
 * @param serverPort
 */
function renderHealthCheckAndWait(
  serverPort: number,
  preCache: boolean,
): string {
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
  sleep 15
done

echo "vLLM is ready"
jq '.status = "running"' "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"

# ═══ JIT CACHE SAVE (background) ═══
# Health check passed → all JIT compilations complete.
# Archive the tmpfs $HOME to shared storage for next cold start.
CACHE_KEY=$(sha256sum "$VLLM_CONFIG" | cut -d' ' -f1)
CACHE_FILE="$PROJECTDIR/ivllm/caches/\${CACHE_KEY}.tar.gz"

(
  # Stop the RAM monitor first — avoids noisy output during tar
  if [ -n "\${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null; wait "$MONITOR_PID" 2>/dev/null
  fi

  echo "[cache-save] Archiving JIT cache to shared storage..."

  # Kill any lingering compilation processes so the tar is clean
  sleep 2

  tar czf "\${CACHE_FILE}.tmp" -C "$LOCALDIR" user_home 2>/dev/null && \\
  mv "\${CACHE_FILE}.tmp" "\$CACHE_FILE" && \\
  echo "[cache-save] Done: $(du -sh "\$CACHE_FILE" | cut -f1)" || \\
  echo "[cache-save] Failed"
) &
disown

${renderWaitBlock(preCache)}

finalize_and_exit $EXIT_CODE "vLLM exit"`;
}
// TODO: does not exist clealy when running interactive

/**
 *
 * @param opts
 */
function renderSingleNodeScript(ss: SessionState): string {
  const opts = ss.startArgs!;
  const paths = ss.paths;

  const runtimePayload = renderSingleNodePayload(ss);

  // Calculate resources using our fractional node logic
  const isFullNode = opts.gpuCount === 4;
  const memValue = isFullNode ? '0' : `${opts.gpuCount * 115}G`;
  // const cpusPerTask = isFullNode ? '256' : `${opts.gpuCount * 64}`;
  const exclusiveFlag = isFullNode ? '#SBATCH --exclusive\n' : '';

  if (opts.isInteractive) {
    // DIRECT INTERACTIVE ACCESS (Via SSH execution wrapper)
    // The script payload itself runs raw because your local orchestrator
    // will invoke this string via an active 'srun' command over SSH.
    return `#!/bin/bash
    # Redirect stdout and stderr to both the console and the log file simultaneously
    exec > >(tee -a "${paths.remoteJobLogFile}") 2>&1
    ${runtimePayload}
    echo "Submitted interactive job $VLLM_PID"
    `;
  } else {
    // BATCH PROCESSING ACCESS (Produces a traditional SBATCH file)
    // return `#!/bin/bash
    // #SBATCH --job-name=${opts.jobName}
    // #SBATCH --nodes=1
    // #SBATCH --gpus=${opts.gpuCount}
    // #SBATCH --mem=${memValue}
    // #SBATCH --cpus-per-task=${cpusPerTask}
    // #SBATCH --time=${opts.timeLimit}
    // ${exclusiveFlag}
    // # Write the runtime execution logic directly below the headers
    // ${runtimePayload}`;

    //#SBATCH --partition=interactive
    //#SBATCH --reservation=interactive

    return `#!/bin/bash
#SBATCH --job-name=${opts.jobName}
#SBATCH --nodes=1
#SBATCH --gpus=${opts.gpuCount}
#SBATCH --mem=${memValue}
#SBATCH --cpus-per-gpu=64
#SBATCH --cpu-bind=cores
#SBATCH --time=${opts.timeLimit}
${exclusiveFlag}

exec > "${paths.remoteJobLogFile}" 2>&1
# Write the runtime execution logic directly below the headers
${runtimePayload}
`;
  }
}

/**
 *
 * @param opts
 */
function renderSingleNodePayload(ss: SessionState): string {
  const opts = ss.startArgs!;
  const paths = ss.paths;
  const model = opts.configYaml.model;

  const lcaseModel = model.split('/').pop()!.toLowerCase();
  const maxJobs = Math.min(opts.gpuCount * 2, 8); // Safe, scalable compiler throttle

  return `umask 0002

JOB_DETAILS="${paths.remoteJobLockFile}"
VLLM_CONFIG="${paths.remoteJobVllmConfigFile}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
SERVER_PORT=${opts.serverPort}
COMPUTE_HOSTNAME=$(hostname)

export MAX_JOBS=${maxJobs}
export TORCHINDUCTOR_PARALLEL_COMPILE_THREADS=${maxJobs}
export FLASHINFER_NVCC_THREADS=${maxJobs}
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300

# Write initialising status — LOCAL already created the file with "pending";
# we overwrite with full details now that we have the SLURM context.
jq -n \\
  --arg status "initialising" \\
  --arg job_name "${opts.jobName}" \\
  --arg slurm_job_id "$SLURM_JOB_ID" \\
  --arg compute_hostname "$COMPUTE_HOSTNAME" \\
  --arg model "${model}" \\
  --argjson server_port ${opts.serverPort} \\
  '{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
    compute_hostname: $compute_hostname, model: $model, server_port: $server_port}' \\
  > "$JOB_DETAILS"

module load brics/nccl gcc-native

${renderWorkDirSetup(paths)}
${NVHPC_PREAMBLE}
source ${paths.remoteProjectVllmVenvActivate}
export HF_HOME=${paths.remoteProjectHfDir}
export HF_HUB_OFFLINE=1
${renderEnvVars(opts.configYaml.env)}
${renderExitDiagnostics()}
${renderExitTrap(false)}
cd "$WORK_DIR"

${renderMonitor()}

# vLLM is launched in the background — model, tensor-parallel-size, and all tuning
# options come from the config file; host and port are infrastructure overrides.

vllm serve \\
  --config "$VLLM_CONFIG" \\
  --host 0.0.0.0 \\
  --port ${opts.serverPort} \\
  --served-model-name "${model}" "${lcaseModel}" "default" "${opts.jobName}" \\
  &
VLLM_PID=$!

echo "APP_PID_MATCH:$VLLM_PID"

${renderHealthCheckAndWait(opts.serverPort, opts.preCache)}`;
}

/**
 *
 * @param opts
 */
function renderMultiNodeScript(ss: SessionState): string {
  const opts = ss.startArgs!;
  const paths = ss.paths;
  const runtimePayload = renderMultiNodePayload(ss);
  const nodeCount = Math.ceil(opts.gpuCount / 4);
  const gpusPerNode = Math.ceil(opts.gpuCount / nodeCount);

  if (opts.isInteractive) {
    // Interactive direct access relies on your local JS script executing the parent allocation:
    // e.g. ssh user@host "srun --nodes=X --gpus-per-node=Y --mem=0 --exclusive bash -s < script.sh"
    return `#!/bin/bash
    exec > >(tee -a "${paths.remoteJobLogFile}") 2>&1
    ${runtimePayload}
    echo "Submitted interactive job $VLLM_PID"
    `;
  } else {
    // Traditional SBATCH batch script generation
    return `#!/bin/bash
#SBATCH --job-name=${opts.jobName}
#SBATCH --nodes=${nodeCount}
#SBATCH --gpus-per-node=${gpusPerNode}
#SBATCH --mem=0
#SBATCH --time=${opts.timeLimit}
#SBATCH --exclusive

exec > "${paths.remoteJobLogFile}" 2>&1
${runtimePayload}
`;
  }
}

/**
 *
 * @param opts
 */
function renderMultiNodePayload(ss: SessionState): string {
  const opts = ss.startArgs!;
  const paths = ss.paths;
  const nodeCount = Math.ceil(opts.gpuCount / 4);
  const gpusPerNode = Math.ceil(opts.gpuCount / nodeCount);

  const rayObjectStoreMemoryGiB = 64;
  const envVars = opts.configYaml.env;
  const envPreamble =
    envVars.length > 0
      ? envVars.map((e) => `export ${e.key}=\\"${e.value}\\"`).join(' && ') +
        ' && '
      : '';
  const cpusPerTask = '256';
  const model = opts.configYaml.model;
  const lcaseModel = model.split('/').pop()!.toLowerCase();

  //TODO: Need to propagate env variables to multi-node setup
  // these need to go to every ray process:
  //   MAX_JOBS: "4"
  //   FLASHINFER_NVCC_THREADS: "4"
  //   TORCHINDUCTOR_PARALLEL_COMPILE_THREADS: "4"
  // however the value 4 needs to be computed based on per gpu number of cores
  // and per CPU memory availble. TBD.

  // Multi node compilation caching:
  // E.g. Node 1 and Node 2 write to independent, clean compilation caches
  // This is achieved by rewriting $HOME for each worker to $LOCALDIR which is node local
  // Each node restores caches from $PROJECTDIR/ivllm/caches

  return `
# Multi-node Ray inference payload
umask 0002

JOB_DETAILS="${paths.remoteJobLockFile}"
VLLM_CONFIG="${paths.remoteJobVllmConfigFile}"
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
SERVER_PORT=${opts.serverPort}
GPUS_PER_NODE=${gpusPerNode}
HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
COMPUTE_HOSTNAME=$HEAD_NODE
HEAD_NODE_IP=$(dig +short $HEAD_NODE)
RAY_PORT=6378

# Write initialising status — LOCAL already created the file with "pending";
# we overwrite with full details now that we have the SLURM context.
jq -n \\
  --arg status "initialising" \\
  --arg job_name "${opts.jobName}" \\
  --arg slurm_job_id "$SLURM_JOB_ID" \\
  --arg compute_hostname "$COMPUTE_HOSTNAME" \\
  --arg model "${model}" \\
  --argjson server_port ${opts.serverPort} \\
  '{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
    compute_hostname: $compute_hostname, model: $model, server_port: $server_port}' \\
  > "$JOB_DETAILS"

module load brics/nccl gcc-native

${renderWorkDirSetup(paths)}
${NVHPC_PREAMBLE}
source ${paths.remoteProjectVllmVenvActivate}
export HF_HOME=${paths.remoteProjectHfDir}
export HF_HUB_OFFLINE=1
${renderEnvVars(envVars)}
${renderMultiNodeExitDiagnostics()}
${renderExitTrap(true)}
RAY_OBJECT_STORE_MEMORY=$((${rayObjectStoreMemoryGiB} * 1024 * 1024 * 1024))

# Required env vars for multi-node Ray+vLLM
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_CROSS_NIC=1
export NCCL_FORCE_FLUSH=0
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1

# Start Ray head node
# bash -c is used to guarantee venv PATH is active on the compute node,
# and to avoid any .local/bin/env shadowing /usr/bin/env on login nodes.
echo "Starting Ray head node ($HEAD_NODE)..."
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --mem=0 \\
  --cpus-per-task=${cpusPerTask} \\
  --ntasks-per-node=1 \\
  bash -c "
    source ${paths.remoteProjectVllmVenvActivate}
    ${envPreamble}
    VLLM_HOST_IP=$HEAD_NODE_IP ray start \\
      --block \\
      --head \\
      --node-ip-address=$HEAD_NODE_IP \\
      --port=$RAY_PORT \\
      --object-store-memory=$RAY_OBJECT_STORE_MEMORY
    " &
sleep 20

# Start Ray worker nodes
WORKER_NODES=$(scontrol show hostnames $SLURM_NODELIST | tail -n+2)
for WORKER in $WORKER_NODES; do
  WORKER_IP=$(dig +short $WORKER)
  echo "Starting Ray worker node: $WORKER ($WORKER_IP)"
  srun --overlap \\
    --nodelist "$WORKER" \\
    --nodes=1 \\
    --gpus=$GPUS_PER_NODE \\
    --mem=0 \\
    --cpus-per-task=${cpusPerTask} \\
    --ntasks-per-node=1 \\
    bash -c "
      source ${paths.remoteProjectVllmVenvActivate}
      ${envPreamble}
      ${renderWorkDirSetup(paths)}
      VLLM_HOST_IP=$WORKER_IP ray start \\
        --block \\
        --address=$HEAD_NODE_IP:$RAY_PORT \\
        --node-ip-address=$WORKER_IP \\
        --object-store-memory=$RAY_OBJECT_STORE_MEMORY
      " &
done
sleep 20

# Verify Ray cluster is ready
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --mem=0 \\
  --ntasks-per-node=1 \\
  bash -c "source ${paths.remoteProjectVllmVenvActivate} && ray status"

# Start vLLM on the head node via srun --overlap (runs within existing job allocation)
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --mem=0 \\
  --ntasks-per-node=1 \\
  bash -c "
    cd ${paths.remoteJobDir}

    source ${paths.remoteProjectVllmVenvActivate}
    ${envPreamble}
    VLLM_HOST_IP=$HEAD_NODE_IP vllm serve \\
      --config $VLLM_CONFIG \\
      --distributed-executor-backend ray \\
      --host 0.0.0.0 \\
      --port ${opts.serverPort} \\
      --served-model-name \"${model}\" \"${lcaseModel}\" \"default\" \"${opts.jobName}\"
  " \\
  &
VLLM_PID=$!

echo "APP_PID_MATCH:$VLLM_PID"

${renderHealthCheckAndWait(opts.serverPort, opts.preCache)}`;
}

/**
 *
 * @param workDir
 */
function renderMultiNodeExitDiagnostics(): string {
  return `${renderExitDiagnostics()}
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
        bash -c '
    RAY_SESSION_DIR=$(readlink -f /local/user/$UID/ray/session_latest 2>/dev/null || true)
    RAY_DEST_LITERAL="'"\$RAY_DESTINATION"'"
    mkdir -p "\$RAY_DEST_LITERAL"
    if [ -n "\$RAY_SESSION_DIR" ] && [ -d "\$RAY_SESSION_DIR/logs" ]; then
      cp -a "\$RAY_SESSION_DIR/logs/." "\$RAY_DEST_LITERAL/" 2>/dev/null || true
      printf "%s\\n" "\$RAY_SESSION_DIR" > "\$RAY_DEST_LITERAL/session_dir.txt"
      else
        printf "%s\\n" "No Ray logs found at /local/user/$UID/ray/session_latest" > "\$RAY_DEST_LITERAL/missing.txt"
        fi
        '; then
        printf "%s\\n" "Finished Ray log archival for $NODE_NAME" >> "$ARCHIVE_STATUS_FILE"
        else
          printf "%s\\n" "Ray log archival srun failed for $NODE_NAME" >> "$ARCHIVE_STATUS_FILE"
          fi
          done
  }`;
}

/**
 *
 * @param opts
 */
export function renderInferenceScript(opts: SessionState): string {
  if (opts.startArgs === undefined)
    throw new Error('SessionState not correctly set up');
  return (
    opts.startArgs.gpuCount > 4
      ? renderMultiNodeScript(opts)
      : renderSingleNodeScript(opts)
  ).trimStart();
}

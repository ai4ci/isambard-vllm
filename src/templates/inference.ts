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
export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64:\${LD_LIBRARY_PATH:-}`;

function renderHealthCheckAndWait(workDir: string, serverPort: number): string {
  return `# Poll /health until vLLM is ready
echo "Waiting for vLLM to become healthy on port ${serverPort}..."
MAX_WAIT=1200
READY=0
for i in $(seq 1 $MAX_WAIT); do
  if curl -sf http://localhost:${serverPort}/health > /dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM process died during startup"
    jq '.status = "failed" | .error = "vLLM process died during startup"' \\
      "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
    exit 1
  fi
  sleep 1
done

if [ $READY -eq 0 ]; then
  echo "Timed out waiting for vLLM after \${MAX_WAIT}s"
  kill $VLLM_PID 2>/dev/null
  jq --arg error "vLLM did not become healthy within \${MAX_WAIT}s" \\
    '.status = "timeout" | .error = $error' \\
    "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
  exit 1
fi

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

exit $EXIT_CODE`;
}

function renderSingleNodeScript(opts: InferenceScriptOptions): string {
  const { jobName, model, vllmVersion, hfHome, configFileName, workDir, serverPort, gpuCount, timeLimit } = opts;
  const venvPath = `$PROJECTDIR/ivllm/${vllmVersion}`;

  return `#!/bin/bash
#SBATCH --job-name=${jobName}
#SBATCH --nodes=1
#SBATCH --gpus=${gpuCount}
#SBATCH --time=${timeLimit}
#SBATCH --exclusive

exec > "${workDir}/${jobName}.slurm.log" 2>&1

JOB_DETAILS="${workDir}/job_details.json"
VLLM_CONFIG="${workDir}/${configFileName}"
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

module load brics/nccl

${NVHPC_PREAMBLE}
source ${venvPath}/bin/activate
export HF_HOME=${hfHome}

# Start vLLM in the background — model, tensor-parallel-size, and all tuning
# options come from the config file; host and port are infrastructure overrides.
srun \\
  --nodes=1 \\
  --gpus=${gpuCount} \\
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

  return `#!/bin/bash
#SBATCH --job-name=${jobName}
#SBATCH --nodes=${nodeCount}
#SBATCH --gpus=${gpuCount}
#SBATCH --time=${timeLimit}
#SBATCH --exclusive

exec > "${workDir}/${jobName}.slurm.log" 2>&1

JOB_DETAILS="${workDir}/job_details.json"
VLLM_CONFIG="${workDir}/${configFileName}"
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

module load brics/nccl

${NVHPC_PREAMBLE}
source ${venvPath}/bin/activate
export HF_HOME=${hfHome}

# Required env vars for multi-node Ray+vLLM
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_RAY_COMPILED_DAG=1
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_SPMD_HEAD=1

# Start Ray head node
echo "Starting Ray head node ($HEAD_NODE)..."
srun \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --cpus-per-task 72 \\
  --ntasks-per-node 1 \\
  env VLLM_HOST_IP=$HEAD_NODE_IP ray start --block --head --node-ip-address=$HEAD_NODE_IP --port=$RAY_PORT &
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
    --cpus-per-task 72 \\
    --ntasks-per-node 1 \\
    env VLLM_HOST_IP=$WORKER_IP ray start --block --address=$HEAD_NODE_IP:$RAY_PORT --node-ip-address=$WORKER_IP &
done
sleep 20

# Verify Ray cluster is ready
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --ntasks-per-node 1 \\
  ray status

# Start vLLM on the head node via srun --overlap (runs within existing job allocation)
export VLLM_HOST_IP=$HEAD_NODE_IP
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --ntasks-per-node 1 \\
  vllm serve \\
  --config "$VLLM_CONFIG" \\
  --distributed-executor-backend ray \\
  --host 0.0.0.0 \\
  --port ${serverPort} \\
  &
VLLM_PID=$!

${renderHealthCheckAndWait(workDir, serverPort)}`;
}

export function renderInferenceScript(opts: InferenceScriptOptions): string {
  return (opts.nodeCount > 1 ? renderMultiNodeScript(opts) : renderSingleNodeScript(opts)).trimStart();
}

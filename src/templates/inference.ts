export interface InferenceScriptOptions {
  jobName: string;
  model: string;
  venvPath: string;
  hfHome: string;
  configFileName: string;
  workDir: string;
  serverPort: number;
  gpuCount: number;
  timeLimit: string;
}

export function renderInferenceScript(opts: InferenceScriptOptions): string {
  const {
    jobName, model, venvPath, hfHome, configFileName,
    workDir, serverPort, gpuCount, timeLimit,
  } = opts;

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

module load cudatoolkit

source ${venvPath}/bin/activate
export HF_HOME=${hfHome}

# Start vLLM in the background — model, tensor-parallel-size, and all tuning
# options come from the config file; host and port are infrastructure overrides.
srun \\
  --nodes=1 \\
  --gpus=${gpuCount} \\
  --ntasks=1 \\
  vllm serve \\
  --config "$VLLM_CONFIG" \\
  --host 0.0.0.0 \\
  --port ${serverPort} \\
  &
VLLM_PID=$!

# Poll /health until vLLM is ready
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

exit $EXIT_CODE
`.trimStart();
}

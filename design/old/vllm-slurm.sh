#!/bin/bash
# DEPRECATED: This script uses a reverse SSH tunnel (COMPUTE -> LOGIN) which does not work
# on Isambard AI — COMPUTE nodes cannot initiate outbound SSH connections.
# The correct approach is a forward tunnel established by LOCAL once the job is running:
#   ssh -L <local_port>:<compute_node>:<server_port> <user>@<login_node>
# See design/requirements.md for the current design.
#SBATCH --job-name=vllm-serve
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --output=out/%x.%j.out

#TODO: make working directory configurable
#

source .venv/bin/activate

export HF_HOME="/projects/public/brics/hf"
export MODEL_PATH="$HF_HOME/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/"
export MODEL_NAME="openai/gpt-oss-120b"
export YAML_CONFIG="${HOME}/vllm_tutorial/GPT-OSS_Hopper.yaml"
export TIKTOKEN_ENCODINGS_BASE="/projects/public/brics/distributed_vllm/etc/encodings"
export TENSOR_PARALLELISM_SIZE=4
export SERVER_ADDRESS=$(dig +short ${HOSTNAME}-hsn0)
echo "SERVING ON $HOSTNAME with TENSOR_PARALLELISM_SIZE=$TENSOR_PARALLELISM_SIZE"
echo "Config file: $YAML_CONFIG"
echo "Nodes: $SLURM_JOB_NUM_NODES; GPUs: $SLURM_GPUS"

VLLM_LOGGING_LEVEL=DEBUG

# === TUNNEL CONFIG ===
SERVER_PORT=8000
TUNNEL_PORT=11434  # Local port you'll connect to from OpenCode
LOGIN_NODE="${HOSTNAME}"  # Or your actual login hostname

# module load brics/nccl
module list

export CC=gcc
export CXX=g++

# === START vLLM SERVER IN BACKGROUND ===
# Use & to background, then capture PID for cleanup
srun \
    --nodes=$SLURM_JOB_NUM_NODES \
    --gpus=$SLURM_GPUS \
    --cpus-per-task 72 \
    --ntasks-per-node 1 \
    vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --load-format safetensors \
    --config $YAML_CONFIG \
    --host 0.0.0.0 \
    --port $SERVER_PORT \
    --max-num-seqs 512 \
    --tensor_parallel_size=$TENSOR_PARALLELISM_SIZE \
    &
VLLM_PID=$!

# === WAIT FOR SERVER TO BE READY ===
echo "Waiting for vLLM server to start on port $SERVER_PORT..."
for i in {1..1200}; do
    if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
        echo "✅ vLLM server is ready"
        break
    fi
    if [ $i -eq 1200 ]; then
        echo "❌ Server failed to start within 1200 seconds"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 2
done

# === ESTABLISH REVERSE SSH TUNNEL ===
echo "Establishing reverse tunnel: compute:$SERVER_PORT -> $LOGIN_NODE:$TUNNEL_PORT -> your_machine:$TUNNEL_PORT"

# Important: Use -N (no remote command), -f (background), -R (reverse)
# -o ServerAliveInterval keeps tunnel alive through idle periods
ssh -N -f -R $TUNNEL_PORT:localhost:$SERVER_PORT \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    $USER@$LOGIN_NODE

if [ $? -eq 0 ]; then
    echo "✅ Tunnel established. Connect locally to http://localhost:$TUNNEL_PORT/v1"
    echo "   Test with: curl http://localhost:$TUNNEL_PORT/v1/models"
else
    echo "❌ Failed to establish tunnel. Check SSH keys and login node access."
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# === KEEP JOB ALIVE WHILE SERVER RUNS ===
# Wait for vLLM process; if it exits, job ends
wait $VLLM_PID
EXIT_CODE=$?

# === CLEANUP (optional but recommended) ===
echo "vLLM server exited with code $EXIT_CODE. Cleaning up tunnel..."
# Kill any ssh processes associated with this tunnel (best-effort)
pkill -f "ssh -N -f -R $TUNNEL_PORT:localhost:$SERVER_PORT" 2>/dev/null

exit $EXIT_CODE

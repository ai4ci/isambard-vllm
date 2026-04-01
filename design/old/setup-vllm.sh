#!/bin/bash
#TODO: make location configurable. make uv installation chek if needed. run checks to make sure vllm installed before exiting.
module load cudatoolkit
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir vllm_tutorial
cd vllm_tutorial/ || exit 127
uv venv --seed --python=3.12
# The version of vllm used is important as it must line up with the CUDA support on the HPC.
srun --gpus=1 --pty bash -c "
    source .venv/bin/activate
    uv pip install -U vllm[flashinfer]==0.15.1 ray[default] \
        --torch-backend=auto \
        --extra-index-url https://wheels.vllm.ai/0.15.1/vllm
    vllm --version
"

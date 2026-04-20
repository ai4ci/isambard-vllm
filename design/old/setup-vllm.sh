# NOTE: This script is OBSOLETE and must not be used.
# It predates the NVIDIA HPC SDK 26.3 approach (ADR-011) and uses:
#   - module load cudatoolkit  (CUDA 12.7 max — insufficient for vLLM 0.15.1+)
#   - cu129 wheels from wheels.vllm.ai  (requires CUDA 12.9+, driver ≥ 575)
#   - user home directory install       (should be $PROJECTDIR/ivllm/ instead)
#
# The correct approach is `ivllm setup` which installs:
#   1. NVIDIA HPC SDK 26.3 → $PROJECTDIR/ivllm/nvhpc/ (CUDA 13.1 compat libs)
#   2. vLLM cu130 wheels  → $PROJECTDIR/ivllm/<version>/ (versioned shared venv)
#
# See design/adr.md (ADR-011) and design/implementation.md (Phase F2) for details.
#!/bin/bash
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

export interface SetupScriptOptions {
  vllmVersion: string;
}

export function renderSetupScript(opts: SetupScriptOptions): string {
  const { vllmVersion } = opts;
  const venvDir = `$PROJECTDIR/ivllm/${vllmVersion}`;
  const nvhpcDir = `$PROJECTDIR/ivllm/nvhpc`;
  const nvhpcRoot = `${nvhpcDir}/Linux_aarch64/26.3`;
  const ldLibPath = [
    `$NVHPC_ROOT/cuda/12.9/compat`,
    `$NVHPC_ROOT/cuda/12.9/lib64`,
    `$NVHPC_ROOT/compilers/lib`,
    `$NVHPC_ROOT/comm_libs/12.9/nccl/lib`,
    `$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib`,
    `$NVHPC_ROOT/math_libs/12.9/lib64`,
    `\${LD_LIBRARY_PATH:-}`,
  ].join(":");

  return `#!/bin/bash
# ivllm-setup version 0.2.9000
#SBATCH --job-name=ivllm-setup
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

set -euo pipefail

exec > "$HOME/.config/ivllm/setup.log" 2>&1
echo "=== ivllm-setup version 0.2.9000 ==="

# Install uv if not already present
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p $PROJECTDIR/ivllm

# Phase A: Install NVIDIA HPC SDK 26.3 cuda_multi (provides CUDA 12.9 + 13.1)
if [ ! -d ${nvhpcDir} ]; then
  echo "=== Installing NVIDIA HPC SDK 26.3 (cuda_multi) ==="
  wget https://developer.download.nvidia.com/hpc-sdk/26.3/nvhpc_2026_263_Linux_aarch64_cuda_multi.tar.gz \\
    -O /tmp/nvhpc.tar.gz
  tar xpzf /tmp/nvhpc.tar.gz -C /tmp
  cd /tmp/nvhpc_2026_263_Linux_aarch64_cuda_multi
  NVHPC_SILENT=true NVHPC_INSTALL_DIR=${nvhpcDir} NVHPC_INSTALL_TYPE=single ./install
  rm -f /tmp/nvhpc.tar.gz
  echo "=== HPC SDK install complete ==="
else
  echo "=== HPC SDK already installed at ${nvhpcDir} — skipping ==="
fi

# Phase B: Install vLLM ${vllmVersion} into versioned venv
if [ ! -d ${venvDir} ]; then
  echo "=== Installing vLLM ${vllmVersion} ==="
  module load gcc-native/14.2
  export NVHPC_ROOT=${nvhpcRoot}
  export LD_LIBRARY_PATH=${ldLibPath}
  # $PROJECTDIR is on a different filesystem from the uv cache (~/.cache/uv);
  # hardlinking across filesystems is not possible so set copy mode explicitly.
  export UV_LINK_MODE=copy
  uv venv ${venvDir} --python 3.12
  source ${venvDir}/bin/activate
  echo "Downloading and installing vLLM ${vllmVersion} wheels (may be slow — large download)..."
  uv pip install vllm==${vllmVersion} \\
    --torch-backend=auto \\
    --extra-index-url https://wheels.vllm.ai/nightly/cu129 \\
    --extra-index-url https://pypi.org/simple/
  echo "uv install complete."
  echo "=== vLLM version ==="
  python -c "import importlib.metadata; print('vllm', importlib.metadata.version('vllm'))"
  echo "IVLLM_SETUP_SUCCESS"
else
  echo "=== vLLM ${vllmVersion} already installed at ${venvDir} — skipping ==="
  echo "IVLLM_SETUP_SUCCESS"
fi
`.trimStart();
}

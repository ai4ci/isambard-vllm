export interface SetupScriptOptions {
  vllmVersion: string;
}

export function renderSetupScript(opts: SetupScriptOptions): string {
  const { vllmVersion } = opts;
  const venvDir = `$PROJECTDIR/ivllm/${vllmVersion}`;
  const nvhpcDir = `$PROJECTDIR/ivllm/nvhpc`;
  const nvhpcRoot = `${nvhpcDir}/Linux_aarch64/26.3`;
  const ldLibPath = [
    `$NVHPC_ROOT/cuda/13.1/compat`,
    `$NVHPC_ROOT/cuda/13.1/lib64`,
    `$NVHPC_ROOT/compilers/lib`,
    `$NVHPC_ROOT/comm_libs/13.1/nccl/lib`,
    `$NVHPC_ROOT/comm_libs/13.1/nvshmem/lib`,
    `$NVHPC_ROOT/math_libs/13.1/lib64`,
    `\${LD_LIBRARY_PATH:-}`,
  ].join(":");

  return `#!/bin/bash
#SBATCH --job-name=ivllm-setup
#SBATCH --nodes=1
#SBATCH --time=02:00:00

set -euo pipefail

exec > "$HOME/.config/ivllm/setup.log" 2>&1

# Install uv if not already present
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mkdir -p $PROJECTDIR/ivllm

# Phase A: Install NVIDIA HPC SDK 26.3 (provides CUDA 13.1 forward compatibility)
if [ ! -d ${nvhpcDir} ]; then
  echo "=== Installing NVIDIA HPC SDK 26.3 ==="
  wget https://developer.download.nvidia.com/hpc-sdk/26.3/nvhpc_2026_263_Linux_aarch64_cuda_13.1.tar.gz \\
    -O /tmp/nvhpc.tar.gz
  tar xpzf /tmp/nvhpc.tar.gz -C /tmp
  cd /tmp/nvhpc_2026_263_Linux_aarch64_cuda_13.1
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
    --extra-index-url https://wheels.vllm.ai/cu130
  echo "uv install complete."
  echo "=== vLLM version ==="
  vllm --version
  echo "IVLLM_SETUP_SUCCESS"
else
  echo "=== vLLM ${vllmVersion} already installed at ${venvDir} — skipping ==="
  echo "IVLLM_SETUP_SUCCESS"
fi
`.trimStart();
}

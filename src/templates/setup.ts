import { dirname } from "path";

export interface SetupScriptOptions {
  venvPath: string;
  vllmVersion: string;
  outputFile: string;
}

export function renderSetupScript(opts: SetupScriptOptions): string {
  const venvParent = dirname(opts.venvPath);
  const { venvPath, vllmVersion, outputFile } = opts;

  return `#!/bin/bash
#SBATCH --job-name=ivllm-setup
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=${outputFile}

set -euo pipefail

module load cudatoolkit

# Install uv if not already present
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

VENV_PARENT=${venvParent}
VENV_PATH=${venvPath}

mkdir -p "$VENV_PARENT"

if [ ! -d "$VENV_PATH" ]; then
  cd "$VENV_PARENT"
  uv venv --seed --python=3.12 .venv
fi

srun \\
  --nodes=1 \\
  --gpus=1 \\
  --ntasks=1 \\
  bash -c "
    source ${venvPath}/bin/activate
    uv pip install -U vllm[flashinfer]==${vllmVersion} ray[default] \\
      --torch-backend=auto \\
      --extra-index-url https://wheels.vllm.ai/${vllmVersion}/vllm
    echo '=== vLLM version ==='
    vllm --version
    echo 'IVLLM_SETUP_SUCCESS'
  "
`.trimStart();
}

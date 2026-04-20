# Architecture Decision Records — isambard-vllm

## ADR-001: LOCAL CLI language — Node.js + bun

**Status**: Accepted

**Context**: The LOCAL CLI needs to manage async processes (SSH tunnel child process, heartbeat timer, stdin for user input), handle SSH subprocess execution, and potentially evolve into a lightweight API routing server.

**Decision**: Implement the LOCAL CLI in Node.js using bun as the runtime and package manager.

**Rationale**:
- Bun provides fast startup, built-in TypeScript support, and a single-binary distribution — good for a CLI tool.
- Node.js has mature primitives for managing child processes (`child_process.spawn`), async I/O, and timers, all needed for the session-owner pattern.
- Aligns with the future routing server direction (an HTTP server is trivial to add).
- LOGIN and COMPUTE scripts remain plain bash — no runtime dependency on the HPC side.

**Consequences**: Requires bun installed on LOCAL. Not natively portable to Windows (out of MVP scope).

---

## ADR-002: Session-owner pattern for `ivllm start`

**Status**: Accepted

**Context**: The vLLM inference session has a clear lifecycle: SLURM job submission → initialisation → running → shutdown. The SSH tunnel and heartbeat must be active for the duration and cleaned up reliably on exit.

**Decision**: `ivllm start` is a long-running foreground process that owns the entire session lifecycle. It does not exit until the session ends.

**Rationale**:
- Keeps tunnel management, heartbeat, and cleanup co-located in one process with straightforward signal handling.
- Avoids the complexity of a background daemon and IPC.
- Natural UX: the terminal running `ivllm start` shows live status; Ctrl+C or typing "exit" cleanly shuts down.
- `ivllm stop` exists purely as a recovery tool for unclean exits.

**Consequences**: The user must keep the terminal open for the duration of the session. A background/detach mode is a future consideration.

---

## ADR-003: `job_details.json` as the SLURM ↔ LOCAL communication channel

**Status**: Accepted

**Context**: LOCAL needs to know when vLLM is ready and what hostname/port to tunnel to. The SLURM job runs on a COMPUTE node with no direct connection back to LOCAL.

**Decision**: The SLURM script writes status updates and connection details to `job_details.json` in the job's working directory on the HPC (parallel filesystem, visible to both LOGIN and COMPUTE). LOCAL polls this file via SSH.

**Schema**:
```json
{
  "status": "pending" | "initialising" | "running" | "failed" | "timeout",
  "job_name": "<job>",
  "slurm_job_id": "<id>",
  "compute_hostname": "<hostname>",
  "server_port": 8000,
  "model": "<model-name>",
  "error": "<optional error message>"
}
```

**Rationale**:
- Simple, no additional infrastructure. The parallel filesystem (`$HOME` or `$PROJECTDIR`) is visible to all nodes.
- `jq` on the HPC makes atomic field updates straightforward in bash.
- Acts as a lockfile: existence of the file prevents duplicate jobs with the same name.

**Consequences**: LOCAL must poll via SSH (small overhead). File must be cleaned up on shutdown — handled by the shutdown sequence in `ivllm start` and as recovery in `ivllm stop`.

---

## ADR-004: Forward SSH tunnel from LOCAL

**Status**: Accepted

**Context**: COMPUTE nodes on Isambard AI cannot initiate outbound SSH connections, ruling out reverse tunnels.

**Decision**: LOCAL establishes a forward SSH tunnel once `job_details.json` reports `status: "running"`:
```
ssh -N -L <local_port>:<compute_hostname>:<server_port> <user>@<login_node>
```
This is spawned as a child process of `ivllm start` and killed as part of the shutdown sequence.

**Rationale**:
- The only viable tunnelling direction given HPC network constraints.
- Spawning as a child process ties tunnel lifetime to the `ivllm start` process.

**Consequences**: LOCAL must have SSH access to LOGIN (prerequisite, out of scope). The `ssh` binary must be available on LOCAL (standard on Linux/macOS).

---

## ADR-005: vLLM installation location on HPC

**Status**: Superseded by ADR-010

**Context**: vLLM must be installed once and reused across inference jobs. The `uv` venv created during setup must be activatable by the SLURM script.

**Decision (original)**: Install vLLM into a `uv` venv at a fixed path under `$HOME` or `$PROJECTDIR`.

**Why superseded**: The GH200 GPU driver on Isambard AI (565.57.01) only supports CUDA 12.7. Recent vLLM builds require CUDA 12.9+. pip installation of nightly vLLM either fails to build dependencies (e.g. `fastsafetensors`) or hits CUDA library version conflicts at runtime. The Isambard support team recommends using container images instead — see ADR-010.

---

## ADR-006: Fixed local port for MVP

**Status**: Accepted

**Context**: Multiple concurrent jobs with auto-assigned ports require a local registry and add complexity around OpenCode configuration.

**Decision**: MVP uses a single fixed local port (default: 11434, overridable with `--local-port`). No local registry in MVP.

**Rationale**:
- Keeps MVP scope minimal and testable.
- Port is parameterised throughout the implementation so the multi-job registry can be added in a future phase without refactoring.

**Consequences**: Running two jobs simultaneously in MVP is not supported (port conflict). Multi-job support is a tracked future phase.

---

## ADR-007: Model pre-download on LOGIN node

**Status**: Accepted

**Context**: vLLM will automatically download a model from HuggingFace if not cached, but this would occur during the SLURM job on a COMPUTE node. This wastes expensive GPU allocation time during download. COMPUTE nodes do have internet access.

**Decision**: `ivllm start` checks the shared HuggingFace cache (`$PROJECTDIR/hf`) on LOGIN before submitting the SLURM job. If the model is not cached, it runs `huggingface-cli download <model>` on LOGIN via SSH, streaming progress to the user. The SLURM script sets `HF_HOME=$PROJECTDIR/hf` so vLLM uses the pre-populated cache.

**Rationale**:
- LOGIN nodes have internet access and are not metered against GPU allocations.
- `$PROJECTDIR/hf` is shared parallel storage — one download serves all project members and all COMPUTE nodes.
- `huggingface-cli` is installed as a dependency of vLLM in the existing venv, so no extra setup is needed.
- `HF_TOKEN` is read from the LOCAL environment and forwarded via the SSH command.

**Consequences**: `ivllm start` requires a `--model` argument (the HuggingFace model ID). The `HF_TOKEN` environment variable must be set on LOCAL for private or gated models. Cache check is a simple directory existence test (`$PROJECTDIR/hf/hub/models--<org>--<name>`); a failed check falls through to `huggingface-cli download`.

---

## ADR-008: vLLM config YAML as single source of truth for model and parallelism options

**Status**: Accepted

**Context**: `ivllm start` originally accepted `--model` and `--tensor-parallel-size` as CLI flags, which duplicated options that can also appear in the vLLM `--config` YAML. Passing conflicting values on both the CLI and in the YAML creates undefined behaviour.

**Decision**: All vLLM serving options (model, tensor-parallel-size, pipeline-parallel-size, max-model-len, etc.) are expressed exclusively in the vLLM config YAML. The `--model` and `--tensor-parallel-size` flags are removed from `ivllm start`. `ivllm start` parses the YAML locally (using `js-yaml`) to extract `model` (for the HuggingFace pre-download) and the parallelism sizes (to set `#SBATCH --gpus`). The SLURM script runs `vllm serve --config <file> --host 0.0.0.0 --port <port>` — the `host` and `port` flags are retained as explicit CLI overrides because they are infrastructure concerns (required for the SSH tunnel to work) rather than model configuration.

**Rationale**:
- A single config file is easier to audit, version, and share than a mix of CLI flags and YAML.
- Eliminates risk of conflicting values (e.g. `--tensor-parallel-size 4` on CLI vs `tensor-parallel-size: 2` in YAML).
- The vLLM YAML format already supports all options; users familiar with vLLM docs can use it directly.

**Consequences**: Users must include `model:` in their YAML config. `tensor-parallel-size` (and `pipeline-parallel-size`) in the YAML are used to derive the SLURM GPU allocation; `--gpus` remains as an explicit CLI override. The `--mock` mode retains `--model` as a CLI flag since it does not use a vLLM config file.

---

## ADR-009: Chat template support out of scope for MVP

**Status**: Accepted

**Context**: vLLM's `--chat-template` option accepts either a file path (a Jinja2 template file) or an inline single-line string. Some older models do not embed a chat template in their `tokenizer_config.json` and require one to be supplied explicitly.

**Decision**: Chat template file copying is out of scope for MVP. The single-line inline form is already supported at no cost (it is a plain YAML value in the config file). File-based templates are not supported — users who need them must copy the file to the HPC manually and reference its remote path in the YAML.

**Rationale**:
- Modern models (Llama 3, Qwen 2.5, Mistral, etc.) embed their chat template in the tokeniser config; vLLM picks it up automatically from the HuggingFace cache.
- The inline single-line form covers the remaining cases without requiring any additional file-copy logic.
- Adding file detection (is the `chat-template:` value a local path?) and an extra `scp` call adds complexity for an edge case that is unlikely to arise in practice on Isambard AI.

**Consequences**: If a user needs a file-based chat template, they must `scp` it to the HPC themselves and set `chat-template: /remote/path/template.jinja` in their YAML. This can be revisited if it becomes a recurring pain point during E2E testing.

---

## ADR-010: Singularity container for vLLM on HPC

**Status**: On hold — superseded by ADR-011. Preserved for future reference; Singularity remains a viable path for single-node and potentially multi-node if bare-metal limitations arise.

**Context**: The GH200 GPU driver on Isambard AI (565.57.01) supports CUDA 12.7 at most. Recent vLLM releases require CUDA 12.9+. pip installation of vLLM nightly fails:
- Build-time: `fastsafetensors` C++ extension fails to compile against system GCC
- Runtime: CUDA library version mismatches (`libnvJitLink`, `libcuda`) even with the forward-compat package

The Isambard support team recommends using Singularity/Apptainer with NGC container images. The official vLLM Docker images (`vllm/vllm-openai:<version>`) ship CUDA forward-compatibility libraries and support `VLLM_ENABLE_CUDA_COMPATIBILITY=1` to activate them automatically.

**Decision**: Replace pip/venv-based vLLM installation with a Singularity image:

1. **`ivllm setup`** submits a CPU-only SLURM job that runs `singularity pull` to convert the Docker image to a `.sif` file. The image is stored in shared project space (`/projects/<project>/ivllm/images/`). This is a one-time team operation, not per-user.

2. **SLURM inference template** runs:
   ```bash
   singularity run --nv \
     --env VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
     <image.sif> vllm serve --config <config> --host 0.0.0.0 --port <port>
   ```
   instead of activating a venv.

3. **Version is specified** in `~/.ivllm/config.yaml` as `vllmImage` (default: `docker://vllm/vllm-openai:latest`). The per-job `vllm.yaml` config may specify `min-vllm-version: X.Y.Z` to reject an image that is too old; `ivllm start` reads the version label from the `.sif` and fails early if the requirement is not met.

4. **Image path** is configurable as `vllmImagePath` in `~/.ivllm/config.yaml`, defaulting to `/projects/<project>/ivllm/images/vllm-openai.sif`. Users who want a private image can point at `~/ivllm/images/` instead.

**Rationale**:
- Official vLLM images include CUDA compat libs; `VLLM_ENABLE_CUDA_COMPATIBILITY=1` handles `LD_LIBRARY_PATH` setup automatically — no manual library wrangling.
- Singularity/Apptainer is Isambard's recommended and supported container runtime for GPU jobs.
- `brics/apptainer-multi-node` module is available on Isambard for multi-node container workloads.
- Shared image in `/projects/` avoids each user pulling a multi-GB image independently.
- `singularity pull` is CPU-intensive and takes several minutes — must run on COMPUTE, not LOGIN.
- Versioned images pin the vLLM version, improving reproducibility; `min-vllm-version` in `vllm.yaml` provides a safety check.

**Consequences**:
- `ivllm setup` now submits a SLURM job rather than running pip install. It no longer has a venv to activate.
- SLURM templates (single-node and multi-node) replace `source .venv/bin/activate && vllm serve` with `singularity run --nv`.
- `ivllm start` validates the `.sif` exists (and meets `min-vllm-version` if specified) before submitting the inference job.
- `ivllm setup` is idempotent: if the `.sif` already exists and matches the configured version, it skips the pull.
- HuggingFace model cache (`$PROJECTDIR/hf`) and job working directories are bind-mounted into the container at runtime (`--bind $PROJECTDIR`).
- `huggingface-cli` used for pre-download (ADR-007) must be available outside the container on LOGIN — this may require a separate lightweight Python install for `huggingface_hub`. Alternatively, pre-download can run inside the container via `singularity exec`.

---

## ADR-011: NVIDIA HPC SDK bare-metal approach for CUDA forward compatibility

**Status**: Accepted (supersedes ADR-010)

**Context**: Two viable options were identified to run vLLM (which requires CUDA 12.9+) on Isambard AI (driver 565.57.01, max CUDA 12.7):

1. **Singularity containers** (ADR-010): clean single-node, but multi-node via Ray requires unproven `source /host/adapt.sh` pattern inside every `srun` task; large image (~10-15 GB); more `ivllm` code changes.
2. **NVIDIA HPC SDK bare-metal**: install the HPC SDK once to shared project space; set `LD_LIBRARY_PATH` to activate CUDA 13.1 forward compatibility; keep existing pip/venv + Ray architecture unchanged.

The Isambard AI documentation now explicitly documents the HPC SDK approach with the exact `LD_LIBRARY_PATH` to use, making it a low-risk, well-supported path. Multi-node Ray remains unchanged.

**Decision**: Use NVIDIA HPC SDK 26.3 (CUDA 13.1) installed once to shared project space. pip-install vLLM into a shared venv. All paths are fixed — no user configuration required.

**Installation layout** (all under `$PROJECTDIR/ivllm/`, managed by `ivllm setup`):

```
$PROJECTDIR/ivllm/
  nvhpc/          ← NVIDIA HPC SDK 26.3 (CUDA 13.1 compat)
  venv/           ← uv venv with vLLM nightly (cu129)
```

**`LD_LIBRARY_PATH` in all SLURM scripts** (set before `vllm serve` or `ray start`):
```bash
export NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3
export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/13.1/compat:$NVHPC_ROOT/cuda/13.1/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/13.1/nccl/lib:$NVHPC_ROOT/comm_libs/13.1/nvshmem/lib:$NVHPC_ROOT/math_libs/13.1/lib64:$LD_LIBRARY_PATH
```

**`ivllm setup`** submits a CPU-only SLURM job that:
1. Downloads the HPC SDK aarch64 tarball (~3 GB) and installs it to `$PROJECTDIR/ivllm/nvhpc/`
2. Sets up the compat `LD_LIBRARY_PATH`
3. Loads `gcc-native/14.2` module (required to compile `fastsafetensors` C++ extension)
4. Creates a uv venv at `$PROJECTDIR/ivllm/venv/` and pip-installs vLLM nightly (cu129)

**Rationale**:
- HPC SDK approach is officially documented by Isambard AI with exact commands and `LD_LIBRARY_PATH`.
- Multi-node Ray architecture is unchanged — just env vars added to SLURM templates.
- Fewer `ivllm` code changes than container approach: only SLURM templates and setup script change.
- Shared project install (`$PROJECTDIR/ivllm/`) means one setup per team, not per user — same benefit as Singularity.
- No configurable paths needed (YAGNI): paths are derived from `$PROJECTDIR` which is always set on Isambard.
- Eliminates library ordering confusion: `compat` is first in `LD_LIBRARY_PATH` → `libcuda.so` from compat shadows the system stub.

**Consequences**:
- `venvPath` and `vllmVersion` removed from `~/.ivllm/config.yaml` (YAGNI — path is fixed as `$PROJECTDIR/ivllm/venv`).
- `ivllm start` pre-flight checks `$PROJECTDIR/ivllm/venv` exists on HPC (unchanged mechanism, new path).
- SLURM templates (single-node and multi-node) prepend the HPC SDK `LD_LIBRARY_PATH` before activating the venv and calling `vllm serve` or `ray start`.
- `ivllm setup` is idempotent: skips HPC SDK install if `$PROJECTDIR/ivllm/nvhpc` already exists; skips venv creation if `$PROJECTDIR/ivllm/venv` already exists (unless `--force` passed).
- `fastsafetensors` build requires `module load gcc-native/14.2` during setup — added to setup SLURM script.
- HuggingFace pre-download (ADR-007) continues to use `huggingface-cli` from the shared venv on LOGIN.

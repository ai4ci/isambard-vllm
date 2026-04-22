# Implementation roadmap — isambard-vllm

## Phase 1 — Project scaffold and tooling

- [x] Initialise bun/Node.js project (`bun init`, `package.json`, TypeScript config)
- [x] CLI entry point with sub-command routing (`ivllm setup | start | status | stop`)
- [x] Configuration file (`~/.config/ivllm/config.json`): HPC login host, HPC username, venv path, default local port
- [x] SSH helper module: run a remote command on LOGIN, copy a file to LOGIN (wraps `ssh`/`scp` child processes)
- [x] Commit

## Phase 2 — `ivllm setup`

- [x] Generate setup SLURM script from template (based on `design/old/setup-vllm.sh`)
- [x] Copy script to LOGIN and submit via `sbatch`
- [x] Poll SLURM job status until complete or failed
- [x] Stream SLURM output log back to LOCAL in real time
- [x] Report success (vLLM version) or failure with log excerpt
- [x] Validate venv exists on HPC after setup
- [x] Commit

## Phase 3 — SLURM inference script

- [x] Write SLURM bash template for single-node vLLM inference
  - Activate venv
  - Write initial `job_details.json` (`status: "initialising"`, node hostname, SLURM job ID)
  - Start vLLM with config file, model, port
  - Poll `/health` until ready; update `job_details.json` (`status: "running"`, server port, model name)
  - On failure/timeout: update `job_details.json` (`status: "failed"` / `"timeout"`)
  - Log to file in job working directory
  - No SSH tunnel logic in SLURM script
- [x] Unit-testable template rendering (job name, config path, model, port substitution)
- [x] Commit

## Phase 4 — `ivllm start` — core session owner

- [x] Pre-flight: check venv exists on HPC; fail early if not
- [x] Model pre-download on LOGIN:
  - [x] Check HuggingFace cache at `$PROJECTDIR/hf` for the requested model
  - [x] If not cached: run `huggingface-cli download <model>` on LOGIN via SSH with `HF_HOME=$PROJECTDIR/hf` and `HF_TOKEN` forwarded; stream progress to user
- [x] Create `job_details.json` on HPC (`status: "pending"`); fail if already exists (lockfile)
- [x] Copy vllm config file and generated SLURM script to LOGIN
- [x] Submit SLURM job via `sbatch`; record SLURM job ID
- [x] Poll `job_details.json` on LOGIN; display status transitions to user
- [x] On `status: "running"`: spawn forward SSH tunnel child process
- [x] Print connection URL to user: `http://localhost:<port>/v1`
- [x] Heartbeat loop: poll `/health` through tunnel on configurable interval
- [x] Shutdown sequence (Ctrl+C, "exit" input, heartbeat failure, or SLURM failure):
  1. `scancel` SLURM job via SSH
  2. Terminate tunnel child process
  3. Remove `job_details.json` from HPC
- [x] Handle SIGINT/SIGTERM to trigger shutdown sequence
- [x] Commit

## Phase 5 — `ivllm status` and `ivllm stop`

- [x] `ivllm status [job]`: SSH to LOGIN, read `job_details.json` for named job (or all job working directories); display status table
- [x] `ivllm stop <job>`: recovery path — `scancel` by SLURM job ID from `job_details.json`, kill any lingering tunnel processes on LOCAL, remove `job_details.json`
- [x] Commit

## Phase 6 — Mock vLLM script and `ivllm start --dry-run`

- [x] Write a mock vLLM SLURM script template (`src/templates/mock-inference.ts`) that:
  - Uses the same template approach as `renderInferenceScript` — submitted as a SLURM batch job
  - Writes `job_details.json` correctly (same schema as real inference script)
  - Serves `/health` and `/v1/models` on the COMPUTE node via a lightweight bash HTTP server
  - Simulates a configurable startup delay before marking status `"running"`
- [x] Add `--mock` flag to `ivllm start`: substitutes `renderMockInferenceScript()` for `renderInferenceScript()`, otherwise identical flow
- [x] Implement `--dry-run` flag on `ivllm start` only:
  - SSH primitives are swapped for dry-run equivalents via a thin wrapper in `start.ts`
  - File copies (`scp`) write to a local temp directory instead of the remote LOGIN node
  - SSH remote commands are printed (with full command text) but not executed
  - SLURM job is not submitted
  - Key output for review: generated SLURM script and vLLM config file in local temp dir
  - Works with both real and mock modes (i.e. `--mock --dry-run` reviews the mock script)
- [x] Run `ivllm start <job> ... --dry-run` and review real inference script + config
- [x] Run `ivllm start <job> ... --mock --dry-run` and review mock SLURM script

## Phase 7 - Mock vLLM remote testing
- [x] End-to-end test: `ivllm start` against mock script, verify tunnel, heartbeat, clean shutdown
- [x] Test heartbeat failure path (mock server exits; verify LOCAL detects and shuts down cleanly)
- [x] Test lockfile behaviour (start same job twice)
- [x] Test `ivllm stop` recovery (simulate unclean exit, verify stop cleans up)
- [x] Commit

## Phase 8 — End-to-end test with real vLLM

- [x] Test `ivllm setup` on Isambard AI (resolve ADR-005 venv path question)
- [x] Confirm COMPUTE node has internet access
- [x] Test `ivllm start` with `Qwen/Qwen2.5-0.5B-Instruct` (lightweight model)
- [x] Verify OpenAI API endpoint accessible on LOCAL via tunnel
- [x] Resolve Unknowns: HuggingFace token, model cache location (`$HF_HOME` in `$PROJECTDIR`?)
- [x] Commit

---

## Post-MVP fixes and features

### Issue #1 — Exit before SLURM schedules

- [x] Moved readline setup to immediately after `sbatch` submission, so typing `exit` works during PENDING and initialising phases (not only after vLLM is running)

### Issue #2 — Launch feedback

- [x] While `job_details.json` shows `"pending"`, poll `squeue -j <id>` and print SLURM queue state (e.g. `PENDING Priority`, `RUNNING None`) so the user knows where they are in the queue
- [x] During `"initialising"`, incrementally tail the SLURM log and prefix each new line with `|` to show vLLM startup progress in real time
- [x] Added `parseSlurmQueueState` / `getSlurmQueueState` to `src/slurm.ts`

### Issue #3 — Multi-node inference via Ray

- [x] `resolveGpuCount` now returns `{ gpuCount, nodeCount }` where `nodeCount = ceil(gpuCount / gpusPerNode=4)`; no longer errors on configs with `pipeline-parallel-size > 1`
- [x] `renderInferenceScript` accepts `nodeCount`; when `nodeCount > 1` generates a Ray cluster bootstrap SLURM script:
  - Sets `#SBATCH --nodes=N`
  - Starts Ray head node and worker nodes via `srun` using `env VLLM_HOST_IP=...`
  - Verifies cluster with `ray status`
  - Runs `vllm serve --distributed-executor-backend ray` on head node via `srun --overlap`
  - Sets required env vars (`VLLM_ALLREDUCE_USE_SYMM_MEM=0`, `VLLM_USE_RAY_COMPILED_DAG=1`, `VLLM_USE_RAY_SPMD_WORKER=1`, `VLLM_USE_RAY_SPMD_HEAD=1`)
  - `compute_hostname` written to `job_details.json` is the Ray head node (tunnel target)
- [x] Added `module load brics/nccl` to single-node template (required for NCCL over Slingshot HSN)
- [x] `start.ts` prints `⚠ Multi-node job: N nodes requested` when applicable
- [x] Updated `generate-vllm-config` skill: multi-node is now supported by `ivllm`

### Skills distribution

- [x] Migrated skill from `.github/skills/` to `skills/` following the `skills-npm` convention
- [x] Added `skills-npm` dev dependency and `prepare` script to `package.json`
- [x] Skill bundled in published npm package via `"files": ["skills"]`

### `generate-vllm-config` skill improvements

- [x] Step 2: check official vLLM recipes page before calculating from scratch
- [x] Expert parallelism (`--enable-expert-parallel`) guidance for MoE models
- [x] FP8 quantization guidance for memory-constrained models on H100
- [x] Tool calling: always include `enable-auto-tool-choice` + `tool-call-parser` for instruct/chat models; parser name table by model family
- [x] Prefix caching recommendation
- [x] Reasoning model parser names documented (`qwen3`, `deepseek_r1`, etc.)
- [x] Example config generated for `Qwen/Qwen2.5-0.5B-Instruct` in `design/example/vllm.yaml`

### README updates

- [x] Removed `--model` from all non-mock examples (model is now in `vllm.yaml`)
- [x] Updated options table; added skill install instructions

---

## Phase F2 — CUDA forward compatibility via NVIDIA HPC SDK

One-time shared install of NVIDIA HPC SDK 26.3 into `$PROJECTDIR/ivllm/nvhpc/`.
Versioned vLLM venvs at `$PROJECTDIR/ivllm/<version>/` (cu130 wheels).
Enables CUDA 13.1 forward compatibility on Isambard AI (driver 565.57.01). See ADR-011.

Path layout (derived from `vllmVersion` config, not separately configurable):
- `$PROJECTDIR/ivllm/nvhpc/` — HPC SDK install (shared, installed once)
- `$PROJECTDIR/ivllm/<vllmVersion>/` — versioned vLLM venv (e.g. `0.19.1/`)

### F2.1 — Config changes

- [x] Remove `venvPath` from `Config` interface in `src/config.ts` (path is now derived from `vllmVersion`)
- [x] Keep `vllmVersion` in `Config` — it determines both the pip install version and the venv directory
- [x] Update `DEFAULTS` and config read/write to remove `venvPath`
- [x] Remove `--venv-path` flag from `ivllm config` command; keep `--vllm-version`
- [x] Write failing tests confirming `venvPath` is absent and `vllmVersion` is present in config schema
- [x] Confirm tests fail, implement, confirm pass

### F2.2 — Setup template

- [x] Rewrite `renderSetupScript` in `src/templates/setup.ts`:
  - Uses `--gpus=1` so `--torch-backend=auto` can detect CUDA via `nvidia-smi` (changed from CPU-only)
  - Phase A — HPC SDK install (skip if `$PROJECTDIR/ivllm/nvhpc` already exists):
    ```bash
    mkdir -p $PROJECTDIR/ivllm
    wget https://developer.download.nvidia.com/hpc-sdk/26.3/nvhpc_2026_263_Linux_aarch64_cuda_multi.tar.gz -O /tmp/nvhpc.tar.gz
    tar xpzf /tmp/nvhpc.tar.gz -C /tmp
    cd /tmp/nvhpc_2026_263_Linux_aarch64_cuda_multi
    NVHPC_SILENT=true NVHPC_INSTALL_DIR=$PROJECTDIR/ivllm/nvhpc NVHPC_INSTALL_TYPE=single ./install
    rm -f /tmp/nvhpc.tar.gz
    ```
  - Phase B — vLLM venv install (skip if `$PROJECTDIR/ivllm/<version>` already exists):
    ```bash
    module load gcc-native/14.2
    export NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3
    export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64:${LD_LIBRARY_PATH:-}
    uv venv $PROJECTDIR/ivllm/<version> --python 3.12
    source $PROJECTDIR/ivllm/<version>/bin/activate
    uv pip install vllm==<version> --torch-backend=auto --extra-index-url https://wheels.vllm.ai/<version>/cu129
    ```
    Note: `cuda_multi` image used (provides both 12.9 and 13.1); `cu129` wheels used (not `cu130`)
  - Emit `IVLLM_SETUP_SUCCESS` marker on completion
- [x] Update `SetupScriptOptions` interface: remove `venvPath`; keep `vllmVersion` as the sole version field
- [x] Write failing tests in `tests/setup.test.ts`:
  - Script contains `nvhpc_2026_263_Linux_aarch64_cuda_multi`
  - Script contains `gcc-native/14.2`
  - Script contains `$PROJECTDIR/ivllm/nvhpc`
  - Script contains the versioned venv path (e.g. `$PROJECTDIR/ivllm/0.19.1`)
  - Script contains `NVHPC_ROOT` and the compat `LD_LIBRARY_PATH` (with `cuda/12.9/compat` first)
  - Script contains `uv pip install vllm==` and `wheels.vllm.ai/<version>/cu129`
  - Script does NOT contain `singularity` or `cu130`
- [x] Confirm tests fail, implement, confirm pass

### F2.3 — Setup command

- [x] Update `src/commands/setup.ts`:
  - Remove `venvPath` reference; pass `vllmVersion` to `renderSetupScript`
  - Success check: `test -d $PROJECTDIR/ivllm/<vllmVersion>/bin` on LOGIN
  - Update console output to show version and install path
- [x] Write failing tests for success check using versioned path
- [x] Confirm tests fail, implement, confirm pass

### F2.4 — Inference templates

- [x] Add `LD_LIBRARY_PATH` preamble and versioned venv activation to `renderInferenceScript`:
  ```bash
  export NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3
  export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/13.1/compat:$NVHPC_ROOT/cuda/13.1/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/13.1/nccl/lib:$NVHPC_ROOT/comm_libs/13.1/nvshmem/lib:$NVHPC_ROOT/math_libs/13.1/lib64:$LD_LIBRARY_PATH
  source $PROJECTDIR/ivllm/<vllmVersion>/bin/activate
  ```
- [x] Update single-node template: replace `venvPath` with `$PROJECTDIR/ivllm/<vllmVersion>`
- [x] Update multi-node template: same preamble before `ray start` calls and `vllm serve`
- [x] Remove `venvPath` from `InferenceScriptOptions`; add `vllmVersion: string`
- [x] Tests: `NVHPC_ROOT`, `cuda/13.1/compat`, versioned venv path, multi-node preamble order
- [x] Tests pass (180 total)

### F2.5 — vllm-config: `min-vllm-version` support

- [x] Add optional `min-vllm-version` field to `VllmConfig` interface in `src/vllm-config.ts`
- [x] `parseVllmConfig` reads and returns `minVllmVersion?: string`
- [x] Tests pass for `parseVllmConfig` with/without `min-vllm-version`

### F2.6 — Start command

- [x] Update `src/commands/start.ts` pre-flight:
  - Replace `config.venvPath` lookup with versioned path: `test -f $PROJECTDIR/ivllm/<vllmVersion>/bin/activate`
  - If `min-vllm-version` set in `vllm.yaml`, compare semver against `config.vllmVersion`; fail early if not satisfied
- [x] Pass `vllmVersion` (not `venvPath`) into `renderInferenceScript`
- [x] `src/semver.ts` extracted for testability; 8 unit tests pass

### F2.7 — Dry-run verification

- [ ] Run `ivllm start <job> --config <file> --dry-run` and verify generated SLURM script:
  - Contains `NVHPC_ROOT` and `LD_LIBRARY_PATH` with compat path first
  - Contains `source $PROJECTDIR/ivllm/<version>/bin/activate`
  - Does NOT contain `singularity` or `cu129`
- [ ] Run `ivllm start <job> --mock --dry-run` and verify mock script is unchanged

### F2.8 — End-to-end testing on Isambard AI

- [ ] Run `ivllm setup` — submit CPU-only job, verify HPC SDK at `$PROJECTDIR/ivllm/nvhpc/`
- [ ] Verify versioned venv at `$PROJECTDIR/ivllm/<version>/` with vLLM installed
- [ ] Run `ivllm start` with `Qwen/Qwen2.5-0.5B-Instruct`
- [ ] Verify CUDA 13.1 forward compat active (check nvidia-smi output in SLURM log)
- [ ] Verify OpenAI API endpoint accessible on LOCAL via tunnel
- [ ] Commit

---

## Phase F2-alt — Singularity container support (on hold)

Preserved for future consideration. See ADR-010 for rationale.
Revisit if bare-metal pip install proves fragile across vLLM version updates,
or if multi-node container support via `brics/apptainer-multi-node` is validated with Ray.

### F2-alt.1 — Config changes

- [ ] Add `vllmImage: string` to `Config` interface (default: `"docker://vllm/vllm-openai:latest"`)
- [ ] Add `vllmImagePath: string` to `Config` interface (default: `"$PROJECTDIR/ivllm/images/vllm-openai.sif"`)
- [ ] Deprecate `venvPath` and `vllmVersion` fields (keep for backward compat but no longer used by setup/start)
- [ ] Update `DEFAULTS` in `src/config.ts`
- [ ] Add `--vllm-image` and `--vllm-image-path` flags to `ivllm config` command
- [ ] Update `src/commands/config.ts` help text

### F2-alt.2 — Setup template

- [ ] Rewrite `renderSetupScript` in `src/templates/setup.ts`:
  - CPU-only SLURM job (no `--gpus`, no GPU allocation needed for `singularity pull`)
  - `module load brics/apptainer-multi-node`
  - Create image directory: `mkdir -p $(dirname <imagePath>)`
  - `singularity pull --force <imagePath> <image>` (note: pull is slow, ~10–20 min)
  - Emit `IVLLM_SETUP_SUCCESS` marker on completion
- [ ] Update `SetupScriptOptions` interface: replace `venvPath`/`vllmVersion` with `vllmImage`/`vllmImagePath`
- [ ] Write failing tests in `tests/setup.test.ts`:
  - Script contains `singularity pull`
  - Script contains `brics/apptainer-multi-node`
  - Script does NOT contain `uv pip install` or `venv`
  - Script contains the image path and image source
- [ ] Confirm tests fail, then implement, then confirm tests pass

### F2-alt.3 — Setup command

- [ ] Update `src/commands/setup.ts`:
  - Pre-flight: check `.sif` exists at `vllmImagePath` → skip if present (idempotent); add `--force` flag to override
  - Pass `vllmImage`/`vllmImagePath` to `renderSetupScript`
  - Success check: `test -f <vllmImagePath>` on LOGIN (replaces venv check)
  - Update console output (remove venv references)
- [ ] Write failing tests in `tests/setup.test.ts` for idempotency and image path check
- [ ] Confirm tests fail, implement, confirm pass

### F2-alt.4 — Inference templates

- [ ] Add `vllmImagePath: string` to `InferenceScriptOptions` in `src/templates/inference.ts`
- [ ] Single-node template: replace `source <venv>/bin/activate && vllm serve ...` with:
  ```bash
  singularity run --nv \
    --env VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
    --bind $PROJECTDIR \
    <vllmImagePath> \
    vllm serve --config <configPath> --host 0.0.0.0 --port <port>
  ```
- [ ] Multi-node template: same substitution for the `srun --overlap` vllm serve call;
  Ray head/worker `srun` calls also need `singularity exec --nv <image> /host/adapt.sh bash -c "ray start ..."` — requires `module load brics/apptainer-multi-node`
- [ ] Write failing tests in `tests/inference.test.ts`:
  - Single-node: contains `singularity run --nv`, `VLLM_ENABLE_CUDA_COMPATIBILITY=1`, `--bind $PROJECTDIR`
  - Single-node: does NOT contain `source` or `activate`
  - Multi-node: vllm serve line and ray start lines use singularity with `/host/adapt.sh`
- [ ] Confirm tests fail, implement, confirm pass

### F2-alt.5 — vllm-config: `min-vllm-version` support

- [ ] Add optional `min-vllm-version` field to `VllmConfig` interface in `src/vllm-config.ts`
- [ ] `parseVllmConfig` reads and returns `minVllmVersion?: string`
- [ ] Add `validateImageVersion(imagePath, minVersion, config)` helper:
  - Runs `singularity inspect --labels <imagePath>` on LOGIN via SSH
  - Parses `org.opencontainers.image.version` label
  - Compares semver; throws if image is too old
- [ ] Write failing tests for `parseVllmConfig` with `min-vllm-version` field
- [ ] Confirm tests fail, implement, confirm pass

### F2-alt.6 — Start command

- [ ] Update `src/commands/start.ts` pre-flight:
  - Replace venv existence check with `.sif` existence check: `test -f <vllmImagePath>`
  - If `min-vllm-version` set in yaml config, call `validateImageVersion` and fail early if not met
- [ ] HuggingFace pre-download via `singularity exec --bind $PROJECTDIR <imagePath> huggingface-cli download ...`
- [ ] Pass `vllmImagePath` from config into `renderInferenceScript`
- [ ] Write failing tests in `tests/start.test.ts` for new pre-flight check
- [ ] Confirm tests fail, implement, confirm pass

### F2-alt.7 — End-to-end testing on Isambard AI

- [ ] Run `ivllm setup` — submit CPU-only job, verify `.sif` created in `/projects/.../ivllm/images/`
- [ ] Run `ivllm start` with `Qwen/Qwen2.5-0.5B-Instruct` using container
- [ ] Verify OpenAI API endpoint accessible on LOCAL via tunnel
- [ ] Commit

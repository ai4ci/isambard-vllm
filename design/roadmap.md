## Future Phases (post-MVP)

### MVP
- as described in [design/mvp-requirements.md]

### Phase F1 — MVP Open issues
- Address github issues

### Phase F2 — CUDA forward compatibility via NVIDIA HPC SDK
See ADR-011.
- [x] Remove `venvPath` from `~/.ivllm/config.yaml`
- [x] Remove `vllmVersion` from `~/.ivllm/config.yaml`; `ivllm setup` now takes version as positional arg; `ivllm start` discovers installed versions at runtime (see ADR-014)
- [x] Rewrite `ivllm setup` SLURM template: download HPC SDK 26.3 to `$PROJECTDIR/ivllm/nvhpc/`, create versioned venv at `$PROJECTDIR/ivllm/<version>/`, pip-install vLLM (cu129 wheels) with `gcc-native`; `chmod g+w $PROJECTDIR/ivllm` for multi-user access (ADR-013)
- [x] Add optional `min-vllm-version` field to per-job `vllm.yaml`; `ivllm start` discovers and selects best installed version satisfying the minimum
- [x] Update single-node and multi-node SLURM inference templates: prepend full HPC SDK preamble (`NVHPC_ROOT`, `CUDA_HOME`, `PATH`, `CPATH`, `LD_LIBRARY_PATH`, `CC`, `CXX`, flashinfer cache symlink, `HF_HUB_OFFLINE=1`) before venv activation and vLLM/Ray invocations
- [x] Store HuggingFace token in `~/.config/ivllm/config.json`; use in model download and setup script (ADR-007 update)
- [x] `UV_CACHE_DIR=$LOCALDIR/uv_cache` to prevent multi-user permission conflicts (ADR-013)
- [x] Update `ivllm start` pre-flight: discover installed versions; enforce `min-vllm-version` if set
- [x] Dry-run verification: review generated SLURM scripts
- [x] Single-node end-to-end test on Isambard AI (Qwen2.5-0.5B-Instruct — passing)

### Phase F2-alt — Singularity container support (on hold)
Preserved as ADR-010 for future consideration (single-node clean versioning; multi-node unproven with Ray).
- Revisit if bare-metal pip install proves fragile across vLLM updates.
- See `design/implementation.md` Phase F2-alt for full implementation plan.

### Phase F2.6 — AI coding assistant integration via scoder

When `ivllm start` reaches running state, offer to launch the user's AI coding assistant with the vLLM endpoint pre-configured. Currently the code prints an opencode snippet; this phase would auto-configure assistants and launch them.

Scoder (if available on PATH) provides network sandboxing via bubblewrap + pasta, isolating the workspace while allowing localhost access via `--llm-port`. The assistant's environment variables are passed into the sandbox. scoder autodetects port 11434 for `--llm-port`, but always specify it explicitly.

Menu-based interaction:
- Display the current working directory clearly (e.g. "Launching in /projects/b6ax/my-project")
- Detect which assistant binary is available (`opencode`, `claude`, `code`) and which are sandboxable via scoder
- Show a menu: "Which assistant? [1] opencode [2] claude [3] copilot [4] scoder opencode [5] scoder claude [6] scoder copilot [0] skip"
- Set the correct env vars for the chosen assistant:
  - **opencode**: write/update `opencode.json` in project directory (config precedence: project overrides global; managed config highest priority)
  - **copilot**: set `COPILOT_PROVIDER_BASE_URL=http://localhost:<port>` + `COPILOT_MODEL=<model>`, plus `ANTHROPIC_BASE_URL=http://localhost:4000`, `ANTHROPIC_API_KEY=ollama`, `CLAUDE_MODEL=meta-llama/<model>:free`
  - **claude code**: set `ANTHROPIC_BASE_URL=http://localhost:<port>` + `ANTHROPIC_API_KEY=ollama` + `CLAUDE_MODEL=meta-llama/<model>:free`
- If scoder selected: launch via `scoder --llm-port <port> <assistant>` in a new tmux window
- If no scoder: launch the assistant directly in a new tmux window with env vars set
- When the assistant session exits, return to the menu (loop)
- `--no-launch` flag to suppress auto-launch (show snippet only)

### ⚠ Multi-node inference via Ray (code complete; E2E debugging in progress)
- `resolveGpuCount` returns `{ gpuCount, nodeCount }` from `pipeline-parallel-size`
- `renderInferenceScript` with `nodeCount > 1` generates a Ray cluster bootstrap SLURM script: `#SBATCH --nodes=N`, Ray head/worker startup via `srun bash -c`, `vllm serve --distributed-executor-backend ray`
- `ivllm start` prints `⚠ Multi-node job: N nodes requested` when applicable
- No manual SLURM setup required — pipeline-parallel-size in the vllm.yaml is sufficient
- Extensive environment fixes applied for Isambard AI multi-node (see Phase F2.9 in implementation.md)
- [ ] 2-node run of Qwen3.5-397B-A17B-FP8 to confirm fully working end-to-end

### Phase F2.9 — Multi-node E2E debugging
See implementation.md Phase F2.9 for full bug list. All known bugs fixed. Awaiting next test run to confirm.

### Phase F3 — Model routing server
- Concept: Run a model router on LOGIN, rather than tunnel each `ivllm` instance to LOCAL.
- Support multiple concurrently running `ivllm` instances on COMPUTE nodes.
- Auto port assignment from configurable range (default 11435–11534) allowing connection to COMPUTE from LOGIN
- LOGIN model router is be a lightweight openai API compatible proxy server listening on e.g. port 11434.
- LOGIN model router port forwarded from LOCAL over ssh. Agent harness connects to LOCAL:11434.
- LOGIN model router maintains registry of `vllm.json` configured models available on isambard and port mapping to COMPUTE nodes if running.
- LOGIN model router provides custom `/model/add`, `/model/delete`, `/model/status`, `/model/start`, `/model/stop`, `/model/log`  endpoints which provides details of configured models, current running status, options to add models with `vllm.json` configuration, or delete model configurations, and ability to start and stop a named model, and ability to inspect vllm logs.
- LOGIN model router provides custom `/provider` endpoint which returns opencode compatible provider configuration based on name of models available
- LOGIN model router provides pass through (routing) implementations of all other vllm supported openai API endpoints based on name of model.
- LOGIN model router maintains heartbeat to running models (not LOCAL)
- LOGIN model router shutdown (Ctrl+C / `exit` on LOGIN node process) closes all COMPUTE nodes.
- LOGIN model router automatically shuts down unused models after (e.g. 15 minute) timeout to free up COMPUTE nodes.
- LOGIN model router automatically starts up models when requested (through model parameter of openai api calls) using cached `vllm.json` config.
- requests to model router during model startup sequence returns a "Starting up <modelname>, please try again in a few minutes"

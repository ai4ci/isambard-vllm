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

When `ivllm start` reaches running state, offer to launch the user's AI coding assistant with the vLLM endpoint pre-configured.

- [x] `src/assistant.ts`: utilities for assistant detection (`binaryExists`, `getAvailableAssistants`, `getScoderAvailable`), config generation (`generateOpencodeConfig`, `generateCopilotEnv`, `generateClaudeEnv`), menu building (`buildAssistantMenuOptions`), launch command generation (`getLaunchCommand`)
- [x] `launchAssistantMenu()` in `src/commands/start.ts`: initial interactive menu loop — displays cwd, detects available assistants, offers direct/scoder launch options, configures assistant env, launches in new tmux window, loops back on exit
- [x] Change directory option `[-1]` — prompts for path, validates existence, updates cwd for all subsequent launches
- [x] `--no-launch` flag — suppresses auto-launch, shows config snippet only
- [x] Exit flow: `0` pressed once shows snippet, pressed again exits cleanly
- [x] Scoder integration: `--llm-port <port>` flag opens localhost in sandbox; env vars passed through; scoder autodetects port 11434 but port is always specified explicitly
- [x] Fallback: if remote tmux fails, falls back to local tmux
- [x] 56 unit tests across 5 test files (assistant.test.ts, launch-assistant.test.ts, assistant-menu.test.ts, f26-integration.test.ts, f26-menu.test.ts)

### Phase F2.6b — Launcher wrapper UX and sbx support

- [x] Replace the flattened assistant menu with a 3-layer flow:
  1. target (`change directory` / `OpenCode` / `Copilot` / `Claude` / `shutdown ivllm`)
  2. wrapper (`none` / `scoder` / `sbx` / `back`)
  3. action (`launch now` / `show command` / `back`)
- [x] Always render a shell-ready copy-paste command, including environment variables, for every wrapper mode
- [x] Keep OpenCode on `OPENCODE_CONFIG_CONTENT` runtime overrides instead of writing `opencode.json`
- [x] Add `sbx` wrapper support using `sbx exec -it -w <cwd> -e ... <sandbox> <agent>`
- [x] Resolve sandboxes by agent + workspace, creating them with `sbx create --name <agent>-<basename>` when absent
- [x] Treat `sbx policy allow network localhost:<port>` as a documented user prerequisite; do not mutate sandbox policy automatically
- [x] Preserve direct launch and scoder auto-launch behaviour after command display

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
- Concept: Run a model router LOCAL.
- Support multiple concurrently running `ivllm` instances on COMPUTE nodes.
- Auto port assignment from configurable range (default 11435–11534) allowing multiple connections to COMPUTE from LOCAL
- LOCAL model router is be a lightweight openai API compatible proxy server listening on e.g. port 11434.
- Agent harness connects to LOCAL:11434.
- LOCAL model router maintains registry of `vllm.json` configured models and port mapping to COMPUTE nodes if running.
- LOCAL model router provides custom `/model/add`, `/model/delete`, `/model/status`, `/model/start`, `/model/stop`, `/model/log`  endpoints which provides details of configured models, current running status, options to add models with `vllm.json` configuration, or delete model configurations, and ability to start and stop a named model, and ability to inspect vllm logs.
- LOCAL model router provides custom `/provider` endpoint which returns opencode compatible provider configuration based on name of models available
- LOCAL model router provides pass through (routing) implementations of all other vllm supported openai API endpoints based on name of model.
- LOCAL model router shutdown (Ctrl+C / `exit`) closes all COMPUTE nodes.
- LOCAL model router automatically shuts down unused models after (e.g. 15 minute) timeout to free up COMPUTE nodes.
- LOCAL model router automatically starts up models when requested (through model parameter of openai api calls) using cached `vllm.json` config.
- requests to model router during model startup sequence returns a "Starting up <modelname>, please try again in a few minutes"

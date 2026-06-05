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

### ⚠️ Phase F2.6 — AI coding assistant integration (DEPRECATED)

**Status:** Completed but **deprecated**. Will be spun off into separate project and rearchitected to use the Phase F3 model router.

**Original concept:** When `ivllm start` reaches running state, offer to launch AI coding assistants with vLLM endpoint pre-configured.

**Completed work:**
- [x] `src/assistant.ts`: utilities for assistant detection, config generation, menu building
- [x] `launchAssistantMenu()` in `src/commands/start.ts`: interactive menu loop
- [x] Change directory option, `--no-launch` flag, exit flow
- [x] Scoder and sbx wrapper support
- [x] 56 unit tests across 5 test files

**Future:** Agent launcher will become a thin client that queries the model router's `/admin/provider` endpoint. See Phase F3 for details.

### ⚠ Multi-node inference via Ray (code complete; E2E debugging in progress)
- `resolveGpuCount` returns `{ gpuCount, nodeCount }` from `pipeline-parallel-size`
- `renderInferenceScript` with `nodeCount > 1` generates a Ray cluster bootstrap SLURM script: `#SBATCH --nodes=N`, Ray head/worker startup via `srun bash -c`, `vllm serve --distributed-executor-backend ray`
- `ivllm start` prints `⚠ Multi-node job: N nodes requested` when applicable
- No manual SLURM setup required — pipeline-parallel-size in the vllm.yaml is sufficient
- Extensive environment fixes applied for Isambard AI multi-node (see Phase F2.9 in implementation.md)
- [x] 2-node run of Qwen3.5-397B-A17B-FP8 to confirm fully working end-to-end (Tested & Validated Successfully)

### Phase F2.9 — Multi-node E2E debugging
See implementation.md Phase F2.9 for full bug list. All known bugs (JIT cache races, umask permissions, and custom all-reduce) are successfully resolved, end-to-end multi-node execution validated.

### Phase F3 — Model routing server

**Status:** Implementation in progress (see `design/router.md` for usage guide).

**Core implementation complete:**
- [x] F3.1 — Project scaffold (Fastify HTTP server, config loader) ✅
- [x] F3.2 — Model registry (CRUD, port pool, state tracking) ✅
- [x] F3.3 — SLURM integration (SSH executor, job manager) ✅
- [x] F3.5 — Admin API (add/remove/start/stop/logs/provider endpoints) ✅
- [x] F3.7 — CLI wrapper (`ivllm router` command) ✅
- [x] Router documentation (`design/router.md`) ✅
- [x] Unit tests (35 tests passing) ✅

**Remaining work:**
- [ ] F3.4 — OpenAI API proxy (implement `/v1/chat/completions` proxy to vLLM)
- [ ] F3.6 — Lifecycle management (lazy startup polling, idle timeout, health checks)
- [ ] F3.8 — Agent integration examples (opencode, Copilot, Claude)
- [ ] F3.9 — End-to-end testing on Isambard AI

**Concept:** HTTP server that acts as an OpenAI API-compatible proxy, managing multiple vLLM instances on Isambard COMPUTE nodes. Designed for agent orchestration scenarios.

**Architecture (MVP):**
- Router runs on user's laptop at `http://localhost:11434`
- Manages SLURM jobs via SSH to Isambard LOGIN
- Agents connect to router, router proxies to vLLM backends on COMPUTE nodes
- Future: router can run on LOGIN node (no SSH timeout)

**Key features:**
- `GET /v1/models` — OpenAI-compatible model discovery
- `POST /v1/chat/completions` — Proxied to vLLM with lazy startup
- Admin API: `/admin/models/*` for model lifecycle management
- Model registry: `~/.config/ivllm/models.json`
- Auto port assignment (11435–11534 pool)
- Idle timeout per model (configurable, -1 = never shutdown)
- Lazy startup: agent request triggers SLURM job submission
- Hard cleanup: router shutdown cancels all managed SLURM jobs

**Implementation phases:**
- [ ] F3.1 — Project scaffold (HTTP server, config loader)
- [ ] F3.2 — Model registry (CRUD, port pool, state tracking)
- [ ] F3.3 — SLURM integration (reuse existing templates, SSH abstraction)
- [ ] F3.4 — OpenAI API proxy (model listing, chat completions)
- [ ] F3.5 — Admin API (add/remove/start/stop/logs/provider endpoints)
- [ ] F3.6 — Lifecycle management (lazy startup, idle timeout, health checks)
- [ ] F3.7 — CLI wrapper (`ivllm router` command)
- [ ] F3.8 — Documentation (API reference, agent integration examples)
- [ ] F3.9 — End-to-end testing (lazy startup, concurrent models, cleanup)

**Known limitations (MVP):**
- 12-hour SSH timeout (requires reconnection or login-node deployment)
- No persistence (router restart loses model state)
- Single-user (no authentication)

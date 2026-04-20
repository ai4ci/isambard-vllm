## Future Phases (post-MVP)

### MVP
- as described in [design/mvp-requirements.md]

### Phase F1 — MVP Open issues
- Address github issues

### Phase F2 — CUDA forward compatibility via NVIDIA HPC SDK
See ADR-011.
- [ ] Remove `venvPath` and `vllmVersion` from `~/.ivllm/config.yaml` (YAGNI — paths fixed under `$PROJECTDIR/ivllm/`)
- [ ] Rewrite `ivllm setup` SLURM template: download HPC SDK 26.3 to `$PROJECTDIR/ivllm/nvhpc/`, create venv at `$PROJECTDIR/ivllm/venv/`, pip-install vLLM nightly with `gcc-native/14.2`
- [ ] Update single-node and multi-node SLURM inference templates: prepend HPC SDK `LD_LIBRARY_PATH` before venv activation and vLLM/Ray invocations
- [ ] Update `ivllm start` pre-flight: check `$PROJECTDIR/ivllm/venv` exists (no config lookup needed)
- [ ] Dry-run verification: review generated SLURM scripts
- [ ] End-to-end test on Isambard AI

### Phase F2-alt — Singularity container support (on hold)
Preserved as ADR-010 for future consideration (single-node clean versioning; multi-node unproven with Ray).
- Revisit if bare-metal pip install proves fragile across vLLM updates.
- See `design/implementation.md` Phase F2-alt for full implementation plan.

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

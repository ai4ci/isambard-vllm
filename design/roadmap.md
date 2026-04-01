# Roadmap — isambard-vllm

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

## Phase 6 — End-to-end testing with mock vLLM

- [ ] Write a mock vLLM SLURM script (based on `design/old/tunnel-test.sh` mock HTTP server) that:
  - Writes `job_details.json` correctly
  - Serves `/health` and `/v1/models` on the COMPUTE node
  - Simulates startup delay
- [ ] End-to-end test: `ivllm start` against mock script, verify tunnel, heartbeat, clean shutdown
- [ ] Test heartbeat failure path (mock server exits; verify LOCAL detects and shuts down cleanly)
- [ ] Test lockfile behaviour (start same job twice)
- [ ] Test `ivllm stop` recovery (simulate unclean exit, verify stop cleans up)
- [ ] Commit

## Phase 7 — End-to-end test with real vLLM

- [ ] Test `ivllm setup` on Isambard AI (resolve ADR-005 venv path question)
- [ ] Validate COMPUTE node internet access (resolve Unknown — determines whether LOGIN pre-download is mandatory or a nice-to-have)
- [ ] Test `ivllm start` with `Qwen/Qwen2.5-0.5B-Instruct` (lightweight model)
- [ ] Verify OpenAI API endpoint accessible on LOCAL via tunnel
- [ ] Resolve Unknowns: HuggingFace token, model cache location (`$HF_HOME` in `$PROJECTDIR`?)
- [ ] Commit

---

## Future Phases (post-MVP)

### Phase F1 — Multiple concurrent jobs
- Local registry (`~/.ivllm/registry.json`) mapping job name → local port + tunnel PID
- Auto port assignment from configurable range (default 11434–11534)
- `ivllm status` reports all running jobs with endpoints

### Phase F2 — OpenCode / agent harness integration
- On tunnel up: auto-update OpenCode config to add/update the provider endpoint
- On shutdown: remove or disable the provider entry

### Phase F3 — Multi-node inference
- Extend SLURM template for multi-node (`--nodes=N`, `srun` across nodes)
- Update `job_details.json` schema for multi-node details

### Phase F4 — Model routing server
- Lightweight OpenAI-compatible proxy on LOCAL routing across multiple running jobs

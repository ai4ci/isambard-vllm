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


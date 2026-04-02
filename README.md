# isambard-vllm (`ivllm`)

A CLI tool for managing vLLM inference jobs on [Isambard AI](https://www.isambard.ac.uk/) HPC from your local machine. It submits SLURM jobs, downloads models on the login node, establishes a forward SSH tunnel, and exposes an OpenAI-compatible API on `localhost` — so you can point any agent harness (e.g. OpenCode) straight at your HPC GPU allocation.

```
http://localhost:11434/v1   ←→   ssh tunnel   ←→   vLLM on COMPUTE node
```

---

## Prerequisites

- **Bun** ≥ 1.3 installed locally (`curl -fsSL https://bun.sh/install | bash`)
- A working SSH connection to the Isambard AI login node, with credentials cached in an SSH agent (key-based auth, no interactive password prompts)
- SLURM and `jq` available on the HPC
- A HuggingFace account and access token (`HF_TOKEN`) for gated models

---

## Installation

```bash
git clone https://github.com/ai4ci/isambard-vllm.git
cd isambard-vllm
bun install
bun link          # makes `ivllm` available on your PATH
```

> If `bun link` doesn't put the binary on your PATH, add `~/.bun/bin` to your shell's `PATH`.

---

## Configuration

Run once to configure your connection details where XXXX is your project ID, and YYYY is your user id:

```bash
ivllm config --login-host <login-node>   # e.g. XXXX.aip2.isambard
ivllm config --username <hpc-username>   # e.g. YYYY.XXXX
ivllm config --venv-path <path>          # default: /home/XXXX/YYYY.XXXX/ivllm-venv/.venv
ivllm config --project-dir <path>        # HPC project dir, e.g. /projects/XXXX
ivllm config --local-port <port>         # default: 11434
ivllm config --vllm-version <version>    # default: 0.15.1
```

Settings are saved to `~/.config/ivllm/config.json`. Run `ivllm config` with no arguments to view current settings.

---

## Quickstart

### 1. Install vLLM on the HPC (one-off)

```bash
ivllm setup
```

This submits a SLURM job on a compute node to install vLLM via `uv` into a virtual environment at `venvPath`. Progress is streamed to your terminal. Takes ~10–20 minutes on first run; skipped automatically if the venv already exists.

### 2. Start an inference session

```bash
ivllm start my-job --config vllm.yaml
```

This will:
1. Check SSH connectivity and that the venv exists
2. Read the model name from `vllm.yaml` and download it to the shared HF cache on the login node (if not already cached)
3. Copy the SLURM script and vLLM config to the HPC
4. Submit the SLURM job and monitor startup
5. Establish a forward SSH tunnel once vLLM is healthy
6. Print the local endpoint and run a heartbeat monitor

```
🚀 OpenAI API endpoint: http://localhost:11434/v1
   Model: Qwen/Qwen2.5-0.5B-Instruct

Type 'exit' + Enter to stop, or press Ctrl+C
```

The process stays in the foreground for the lifetime of the session. Press **Ctrl+C** or type **`exit`** to cleanly cancel the SLURM job, close the tunnel, and remove the lockfile.

#### `ivllm start` options

| Flag | Description | Default |
|------|-------------|---------|
| `--config <file>` | vLLM config YAML (contains model, parallelism and all serving options) | required |
| `--local-port <n>` | Local port to expose the API on | from `ivllm config` |
| `--gpus <n>` | GPUs to request (overrides `tensor-parallel-size × pipeline-parallel-size` from YAML) | derived from YAML |
| `--time <hh:mm:ss>` | SLURM time limit | `4:00:00` |
| `--mock` | Use mock vLLM server (no GPU needed — for testing); requires `--model` | off |
| `--dry-run` | Preview generated scripts and scp commands without running anything | off |

### 3. Check job status

```bash
ivllm status           # all known jobs
ivllm status my-job    # specific job
```

### 4. Stop a job (recovery)

If `ivllm start` exits uncleanly (e.g. terminal closed), use:

```bash
ivllm stop my-job
```

This cancels the SLURM job, kills any lingering tunnel process, and removes the lockfile so the job name can be reused.

---

## vLLM config file

Pass any vLLM server options via a YAML config file (see the [vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)). The config file specifies the model and all serving parameters. Example `vllm.yaml`:

```yaml
model: Qwen/Qwen2.5-7B-Instruct
tensor-parallel-size: 1
max-model-len: 32768
gpu-memory-utilization: 0.90
dtype: bfloat16
enable-auto-tool-choice: true
tool-call-parser: hermes
enable-prefix-caching: true
```

### Generating a config with AI

`ivllm` ships an [Agent Skill](https://agentskills.io) for generating `vllm.yaml` files. If you are using an AI coding agent (Cursor, Claude, Windsurf, etc.), the skill will help the agent generate an optimised config for any HuggingFace model on Isambard AI hardware.

**Install the skill** using [skills-npm](https://github.com/antfu/skills-npm). Add `isambard-vllm` as a dependency in your project and run `skills-npm`:

```bash
# In your project directory (where isambard-vllm is a dependency)
bun add isambard-vllm
bunx skills-npm
```

Or if you have cloned this repo and used `bun link`:

```bash
cd /path/to/isambard-vllm && bun link
cd /your/project && bun add isambard-vllm && bunx skills-npm
```

Once installed, ask your AI agent: *"Generate a vllm.yaml config for `Qwen/Qwen2.5-72B-Instruct`"*

---

## HuggingFace token

For gated models, export your token before running `ivllm start`:

```bash
export HF_TOKEN=hf_...
ivllm start my-job --config vllm.yaml
```

The token is forwarded to the login node only for the download step; it is not written to disk or embedded in any generated script.

---

## How it works

```
LOCAL                          LOGIN node                    COMPUTE node
------                         ----------                    ------------
ivllm start
  │─── ssh: mkdir, lockfile ──▶│
  │─── scp: script + config ──▶│
  │─── ssh: sbatch ───────────▶│──── SLURM job ────────────▶│
  │                            │                             │ vLLM starts
  │◀── ssh poll: job_details ──│◀── writes job_details.json─│
  │                            │
  │ (status: running)
  │─── ssh -L localPort:computeHost:serverPort ────────────▶│
  │
  └─── heartbeat: GET http://localhost:localPort/health
```

- The SLURM script writes `job_details.json` into the job working directory (`$HOME/<job>/`) as it progresses through states: `pending → initialising → running → (failed|timeout)`
- `ivllm start` polls this file via SSH to track status and extract the compute hostname for tunnelling
- All tunnelling is initiated by LOCAL; compute nodes cannot initiate outbound SSH connections on Isambard AI

---

## Dry run

Preview what `ivllm start` would do without connecting to the HPC:

```bash
ivllm start my-job --config vllm.yaml --dry-run
```

The generated SLURM script and config file are saved to a local temp directory for inspection. All SSH and scp commands are printed but not executed.

---

## Development

```bash
bun test          # run all tests
bun run start     # run CLI directly
```

Tests use TDD — all tests are in `tests/` and cover config, job argument parsing, SLURM script generation, and remote operation dry-run behaviour.


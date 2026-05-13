# isambard-vllm (`ivllm`)

A CLI tool for managing vLLM inference jobs on [Isambard AI](https://www.isambard.ac.uk/) HPC from your local machine. It submits SLURM jobs, downloads models on the login node, establishes a forward SSH tunnel, and exposes an OpenAI-compatible API on `localhost` — so you can point any agent harness (e.g. OpenCode) straight at your HPC GPU allocation.

```
http://localhost:11434/v1   ←→   ssh tunnel   ←→   vLLM on COMPUTE node
```

---

## Prerequisites

- **Bun** ≥ 1.3 installed locally (`curl -fsSL https://bun.sh/install | bash` or `brew tap oven-sh/bun;
brew install bun` on macOS with homebrew installed)
- A working SSH connection to the Isambard AI login node, with credentials cached in an SSH agent (key-based auth, no interactive password prompts)
- SLURM and `jq` available on the HPC
- A HuggingFace account and access token for gated models (stored via `ivllm config --hf-token`). Hugging Face access token can be created from the [Access Token](https://huggingface.co/settings/tokens) page

---

## Installation

```bash
git clone https://github.com/ai4ci/isambard-vllm.git
cd isambard-vllm
bun install
bun link          # makes `ivllm` available on your PATH
```

> If `bun link` doesn't put the binary on your PATH, add `~/.bun/bin` to your shell's `PATH`.
<details>
<summary>
Click to see how to do so
</summary>
   
> **zsh**
> ```zsh
> echo 'export PATH="$HOME/.bun/bin:$PATH"' >> ~/.zshrc
> source ~/.zshrc
> ```
>
> **bash**
> ```bash
> echo 'export PATH="$HOME/.bun/bin:$PATH"' >> ~/.bashrc
> source ~/.bashrc
> ```
> 
</details>

---

## Configuration

Run once to configure your connection details where XXXX is your project ID, and YYYY is your user id:

```bash
ivllm config --login-host <login-node>   # e.g. XXXX.aip2.isambard
ivllm config --username <hpc-username>   # e.g. YYYY.XXXX
ivllm config --project-dir <path>        # HPC project dir, e.g. /projects/XXXX
ivllm config --local-port <port>         # default: 11434
ivllm config --hf-token <token>          # HuggingFace token for gated models
```

Settings are saved to `~/.config/ivllm/config.json`. Run `ivllm config` with no arguments to view current settings.

---

## Quickstart

### 1. Install vLLM on the HPC (one-off)

```bash
ivllm setup 0.19.1
```

This submits a SLURM job on a compute node to install the NVIDIA HPC SDK 26.3 (providing CUDA 12.9 forward compatibility) and the specified vLLM version into a shared versioned directory at `$PROJECT_DIR/ivllm/0.19.1/`. Progress is streamed to your terminal. Takes ~10–20 minutes on first run; skipped automatically if that version is already installed. To install a different version run `ivllm setup <version>` again.

### 2. vLLM config file

The LLM server is configured via a YAML config file (see the [vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)). The config file specifies the model and all serving parameters. Example `vllm.yaml`:

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

This file needs to be saved locally, and passed in the `--config` parameter. See later
for details on how to create this file.

Ready-to-use example configs for popular models are in the [`examples/`](examples/) directory:

| File | Model | Notes |
|------|-------|-------|
| [`qwen2.5-instruct.yaml`](examples/qwen2.5-instruct.yaml) | Qwen/Qwen2.5-0.5B-Instruct | Dense 0.5B, single node (minimal example) |
| [`qwen3.6-35b-a3b.yaml`](examples/qwen3.6-35b-a3b.yaml) | Qwen/Qwen3.6-35B-A3B | Hybrid MoE 35B, reasoning, single node |
| [`qwen3.5-long-context.yaml`](examples/qwen3.5-long-context.yaml) | Qwen/Qwen3.5-35B-A3B | Hybrid MoE 35B, long context, single node |
| [`gemma-4-31B-it.yaml`](examples/gemma-4-31B-it.yaml) | google/gemma-4-31B-it | Dense 31B multimodal, single node |
| [`gpt-oss-120b.yaml`](examples/gpt-oss-120b.yaml) | openai/gpt-oss-120b | MoE 117B MXFP4, single node |
| [`nemotron-3-super-120B-A12B-BF16.yaml`](examples/nemotron-3-super-120B-A12B-BF16.yaml) | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | Dense 120B reasoning/tool model, single node, requires shared parser plugin |
| [`minimax-m2.5.yaml`](examples/minimax-m2.5.yaml) | MiniMaxAI/MiniMax-M2.5 | MoE 230B, multi-node |

### 3. Start an inference session

```bash
ivllm start my-job --config examples/qwen2.5-instruct.yaml
```

This will:
1. Check SSH connectivity and that the venv exists
2. Read the model name from `vllm.yaml` and download it to the shared HF cache on the login node (if not already cached)
3. Copy the SLURM script and vLLM config to the HPC
4. Submit the SLURM job and monitor startup
5. Establish a forward SSH tunnel once vLLM is healthy
6. Print the local endpoint and run a heartbeat monitor

**N.B. Starting up even a simple model can take a few minutes.**

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
| `--no-launch` | Skip assistant launch menu, show config snippet only | off |

### 4. Launch an AI coding assistant

After starting vLLM, `ivllm start` offers to launch your AI coding assistant with the endpoint pre-configured. When the menu appears:

- **Layer 1 — target**: choose **OpenCode**, **GitHub Copilot**, **Claude Code**, change directory, show the OpenCode config snippet, or shut down `ivllm`
- **Layer 2 — wrapper**: choose **direct launch**, **scoder**, or **sbx** (only wrappers available on your machine are shown)
- **Layer 3 — action**: choose **launch now** or **show copy-paste command**

For every wrapper, `ivllm` prints the full shell-ready command before launching so you can copy, paste, and tweak it manually if needed.

#### sbx prerequisite

If you launch through **sbx**, Docker Sandboxes must be allowed to reach the host-side `ivllm` endpoint first:

```bash
sbx policy allow network localhost:11434
```

Replace `11434` with your configured local port if different. `ivllm` does **not** edit global `sbx policy` rules automatically.

> **Manual configuration (legacy):** If you prefer to configure your assistant manually, add `opencode.json` to your project directory:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "isambard-vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Isambard vLLM Server",
      "options": {
        "baseURL": "http://localhost:11434/v1",
        "apiKey": "EMPTY"
      },
      "models": {
        "Qwen/Qwen2.5-0.5B-Instruct": {
          "name": "Qwen2.5-0.5B-Instruct (Isambard)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  }
}
```

Start up opencode and select the Qwen model from the `Isambard vLLM Server provider`.

When you have finished type "exit" at the terminal you started `ivllm` in and the isambard job will finish.

### 5. Check job status

Not generally necessary as the local session from `ivllm start` will display the current
status.

```bash
ivllm status           # all known jobs
ivllm status my-job    # specific job
```

### 6. Stop a job (recovery)

If `ivllm start` exits uncleanly (e.g. terminal closed), use:

```bash
ivllm stop my-job
```

This cancels the SLURM job, kills any lingering tunnel process, and removes the lockfile so the job name can be reused.

---

## Generating a config with AI

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

For gated models, store your token in the ivllm config:

```bash
ivllm config --hf-token hf_...
```

The token is saved to `~/.config/ivllm/config.json`. It is forwarded to the login node during the model download step and embedded in the setup SLURM script so the HPC can authenticate to HuggingFace. It is not stored in any shared or world-readable location. If not set, `ivllm` falls back to the `HF_TOKEN` environment variable.

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

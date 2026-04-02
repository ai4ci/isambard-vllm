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

**Status**: Proposed (pending validation on HPC)

**Context**: vLLM must be installed once and reused across inference jobs. The `uv` venv created during setup must be activatable by the SLURM script.

**Decision**: Install vLLM into a `uv` venv at a fixed, well-known path — proposed: `$HOME/ivllm-venv/` (or `$PROJECTDIR/ivllm-venv/` if shared across project members is desirable).

**Rationale**:
- `$HOME` is on parallel storage, visible to both LOGIN and COMPUTE.
- A fixed path means the SLURM script can unconditionally `source $HOME/ivllm-venv/.venv/bin/activate` without dynamic discovery.
- `$PROJECTDIR` is preferred if multiple users in the same project share the installation; reduces setup overhead.

**Open question**: Should vLLM be installed under `$HOME` (per-user) or `$PROJECTDIR` (per-project)? Deferred until first HPC test.

**Consequences**: `ivllm setup` must record the venv path. `ivllm start` must validate the venv exists before submitting the SLURM job.

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

**Context**: vLLM will automatically download a model from HuggingFace if not cached, but this would occur during the SLURM job on a COMPUTE node. This wastes expensive GPU allocation time during download and may fail if COMPUTE nodes lack outbound internet access (unknown for Isambard AI).

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

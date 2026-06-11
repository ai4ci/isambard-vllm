# Known Issues — isambard-vllm

## ISSUE-001: DeepSeek-V4-Pro TP=16 fails FP8 block-shape validation

**Status**: Workaround documented — awaiting upstream vLLM fix

**Severity**: High — blocks the recommended 4-node TP=16 configuration

**Affected config**: `examples/deepseek-v4-pro.yaml` (previously defaulted to `tensor-parallel-size: 16`)

### Symptom

vLLM job fails during model weight loading with:

```
ValueError: Weight input_size_per_partition = 192 is not divisible by weight quantization block_k = 128.
```

Full traceback originates in `EngineCoreProc` → `RayWorkerProc.initialize_worker()` → `load_model()` → `validate_fp8_block_shape()` at:
```
vllm/model_executor/layers/quantization/utils/fp8_utils.py:1156
```

### Root Cause

DeepSeek-V4-Pro has `moe_intermediate_size = 3072` and ships with FP8 block-quantised weights (`block_k = 128`, `scale_fmt = ue8m0`). With `tensor_parallel_size = 16`:

```
moe_intermediate_size / tp = 3072 / 16 = 192
192 % 128 = 64  →  not aligned
```

The `down_proj` layer in the shared expert MLP (`DeepseekV4MLP` → `RowParallelLinear`) partitions along the intermediate dimension, producing a per-rank input size of 192 that is not a multiple of the quantisation block size.

This is a vLLM validation constraint — the model's quantised weights were produced at full size, and vLLM does not currently pad or re-align them during tensor-parallel load.

### Divisibility analysis

For TP to work with this model, three constraints must hold simultaneously:

| Constraint | Formula | TP=8 | TP=12 | TP=16 |
|---|---|---|---|---|
| hidden_size divisible by TP | 7168 % tp == 0 | ✅ 896 | ❌ 597.33 | ✅ 448 |
| moe_intermediate_size divisible by TP | 3072 % tp == 0 | ✅ 384 | ✅ 256 | ✅ 192 |
| FP8 block alignment | (3072 / tp) % 128 == 0 | ✅ 3 | ✅ 2 | ❌ 1.5 |

**Only TP ∈ {1, 2, 4, 8}** satisfies all three. TP=16 fails on constraint 3; TP=12 fails on constraint 1.

### Workaround

**TP=8 + PP=2** (16 GPUs, 4 nodes):

- `tensor-parallel-size: 8` — satisfies all three constraints
- `pipeline-parallel-size: 2` — 61 layers split into ~31 + 30 per stage
- Same 16-GPU memory profile as TP=16 (~50 GB weights/GPU)
- ivllm already supports `pipeline-parallel-size` in YAML parsing and GPU resolution

Updated default in `examples/deepseek-v4-pro.yaml`. The TP=16 option remains as a commented alternative.

### Upstream tracking

| Item | Link | Status |
|---|---|---|
| Bug report | [vllm-project/vllm#42384](https://github.com/vllm-project/vllm/issues/42384) | Open (7 comments) |
| Fix PR | [vllm-project/vllm#41312](https://github.com/vllm-project/vllm/pull/41312) | Open (7 comments), not merged |

### Fix approach (PR #41312)

The PR implements **load-time intermediate-size padding**:

1. Pads `moe_intermediate_size` from 3072 → 4096 (smallest multiple of `TP × lcm(block_n, block_k)` = `16 × 256 = 4096`)
2. Zero-pads the affected `gate_proj`, `up_proj`, `down_proj` weights and their `weight_scale_inv` tensors at load time
3. Uses **alternating pad distribution** across TP ranks to avoid leaving some ranks with entirely-zero shared-expert shards
4. SwiGLU preserves zero-in → zero-out, so no activation mask is needed

Padding is only applied when:
- `quant_config` is `Fp8Config` with `weight_block_size` set, AND
- `tp_size × lcm(block_n, block_k)` does not already divide the original size

This mirrors an existing fix for EXAONE4-32B-FP8 ([#34408](https://github.com/vllm-project/vllm/issues/34408)) and addresses the same class of bug reported for Qwen3-Coder-Next-FP8 ([#36853](https://github.com/vllm-project/vllm/issues/36853)).

### Action items

- [x] Update `examples/deepseek-v4-pro.yaml` default to TP=8 + PP=2
- [x] Document the issue and workaround in this file
- [ ] Monitor PR #41312 for merge — revert to TP=16 default once the fix lands
- [ ] Consider backporting the padding fix to our vLLM 0.22.0 install if the upstream PR stalls

---

## ISSUE-002: `ivllm setup` slow — uv falls back from hardlinks to full copy

**Status**: Known, low priority — slow but one-off, not broken

**Severity**: Low — only affects the one-time setup per vLLM version

**Related**: ADR-013 (Multi-user project space permissions)

### Symptom

During `ivllm setup`, uv's package preparation step takes ~30s+ and emits:

```
Prepared 204 packages in 31.73s
warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
         If the cache and target directories are on different filesystems, hardlinking may not be supported.
         If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
```

### Root Cause

`UV_CACHE_DIR` is set to `$LOCALDIR/uv_cache` (RAM-backed tmpfs per ADR-013), but the target venv lives at `$PROJECTDIR/ivllm/<version>/` (Lustre). uv prefers hardlinks for speed, but hardlinks only work within a single filesystem, so it falls back to per-file copies to Lustre — slow for 200+ packages.

ADR-013 deliberately chose `$LOCALDIR` for the cache to avoid multi-user permission conflicts in shared `$PROJECTDIR`. However, the permission concern (setting `chmod g+w` on the cache directory) may already be handled by the broader project space setup, and the hardlink optimisation was knowingly sacrificed.

### Potential fixes

| Approach | Pros | Cons |
|---|---|---|
| Move `UV_CACHE_DIR` to `$PROJECTDIR/ivllm/uv_cache` (Lustre) | Hardlinks work; install near-instant; cache persists across jobs | Need to verify multi-user `chmod g+w` is still safe; cache lives on Lustre |
| Install to `$LOCALDIR`, then rsync/tar to `$PROJECTDIR` | Fast install (hardlinks on RAM disk) | Python venvs are not relocatable — absolute paths in `pyvenv.cfg`, `bin/activate`, `.pth` files break after copy |
| Suppress warning (`export UV_LINK_MODE=copy`) | Cleaner output | No speed improvement |
| Do nothing | It works, one-off cost | ~30s-1min per setup run |

### Recommendation

The simplest fix is moving `UV_CACHE_DIR` to `$PROJECTDIR/ivllm/uv_cache`. The `chmod g+w` already exists for `$PROJECTDIR/ivllm/` (ADR-013), so extending it to the cache directory is a one-liner. This needs verification that multi-user hardlink behaviour on Lustre works correctly (uv hardlinks from cache → venv, both on Lustre).

### Action items

- [ ] Verify current multi-user permissions status on `$PROJECTDIR/ivllm/`
- [ ] Test moving `UV_CACHE_DIR` to `$PROJECTDIR/ivllm/uv_cache` with `chmod g+w`
- [ ] Measure install time improvement

---

## ISSUE-004: DeepSeek-V4-Pro worker hang — `TimeoutError: RPC call to sample_tokens timed out`

**Status**: Open — no known workaround

**Severity**: High — causes random engine death after successful serving

**Affected config**: `examples/deepseek-v4-pro.yaml` (TP=8 + PP=2, vLLM 0.22.1)

### Symptom

vLLM server starts successfully, accepts requests and returns valid responses for several minutes, then a worker silently hangs inside `sample_tokens` and the EngineCore dies after 300s timeout:

```
TimeoutError: RPC call to sample_tokens timed out
RuntimeError: EngineCore encountered a fatal error: TimeoutError
EngineCore encountered a fatal error. Shutting down the engine.
```

**Timeline (run 5, 2026-06-10):**

| Time | Event |
|---|---|
| 12:48–12:59 | Healthy serving — multiple 200 OK responses |
| 12:56:31 | NCCL warning: "unbatched P2P op... new 2-rank NCCL communicator" |
| 12:56:32 | JIT kernel compilation during inference (4 kernels compiled on-the-fly) |
| **13:03:53** | **Timeout fires (~7 min total, ~5 min for RPC)** |

**Previous runs with same crash:**
- Run 4: hung during file-read request, timeout after ~5 min
- Run 5: hung during directory-list request, NCCL+JIT preceded timeout

In all cases the worker process stays alive (`is_alive=True, exitcode=None`) — it does not crash, it just stops responding.

### Root Cause

Silent worker hang inside the `sample_tokens` RPC path. Not a vLLM configuration issue — it is an **open upstream bug** (#41530) with no known root cause or fix.

Key observations from upstream issue #41530:
- Worker stays alive but never sends a response back
- **Not specific to MTP/speculative decoding** — also occurs at TP=2, TP=4
- Affects multiple models: DeepSeek-V4-Pro, Kimi-K2.6, GLM-5.1, others
- Happens in `MultiprocExecutor.get_response` → `MessageQueue.dequeue` → spinloop timeout
- No OOM, no CUDA error, no NCCL error logged at the hang point

**Possible contributing factors** (hypothetical, unproven):
- JIT kernel compilation during inference may introduce concurrency races
- NCCL P2P communicator creation overlapped with inference (12:56:31 warning)
- Prefix caching — disabled in config but did not prevent the hang in run 5
- DeepSeek-V4's MoE routing under sustained traffic

### Status after mitigation

Partial improvement: disabling CUDA graphs (`enforce-eager: true`) and reducing `max-model-len` from 800,000 → 512,000 reduced crash frequency in run 5. The job served multiple requests successfully across several minutes before the timeout. However, the hang still occurs — it is triggered less frequently but not eliminated.

### Workaround

**Partial.** Combining `enforce-eager: true` with reduced `max-model-len` reduces frequency but does not eliminate the hang.

Potential mitigations (untested):
- Set `CUDA_LAUNCH_BLOCKING=1` to serialize GPU operations — would slow inference but might prevent the race
- Reduce `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` from 300 to detect hangs sooner
- `--generation-config vllm` to override model's generation_config defaults
- Lower `gpu-memory-utilization` further (currently 0.88) to reduce memory pressure

### Upstream tracking

| Item | Link | Status |
|---|---|---|
| Bug report | [vllm-project/vllm#41530](https://github.com/vllm-project/vllm/issues/41530) | Open (6 comments), no fix |
| Related: PP+TP timeout | [vllm-project/vllm#38976](https://github.com/vllm-project/vllm/issues/38976) | Open |
| Related: Kimi-K2.6 hang | [vllm-project/vllm#42363](https://github.com/vllm-project/vllm/issues/42363) | Open |
| Related: GLM-5.1 hang | [vllm-project/vllm#40926](https://github.com/vllm-project/vllm/issues/40926) | Open |

### Action items

- [x] Update `examples/deepseek-v4-pro.yaml`: set `enforce-eager: true`, `max-model-len: 512000`
- [x] Update `examples/deepseek-v4-pro.yaml`: disabled `enable-prefix-caching` (no effect on this issue)
- [ ] Monitor upstream issue #41530 for root cause identification
- [ ] Try `CUDA_LAUNCH_BLOCKING=1` as a test to confirm if it's a GPU race condition
- [ ] Try `--generation-config vllm` to use vLLM defaults instead of model's generation_config
- [ ] Consider reporting our GH200-specific observations (NCCL P2P warnings + JIT before hang) to the upstream issue

---

## ISSUE-005: `ivllm start` kills healthy vLLM when local tunnel port is occupied

**Status**: Fixed in code — pre-flight port check added

**Severity**: High — kills healthy jobs with SIGTERM, wastes queue time and SUs

**Affected component**: `src/commands/start.ts`, `src/ssh.ts`

### Symptom

When `ivllm start` is run while port 11434 is already in use by a stale SSH tunnel from a previous session:
1. vLLM starts correctly on the compute node
2. `spawnTunnel` fails to bind local port 11434
3. Tunnel exit handler triggers `shutdown()` → sends `scancel` to the SLURM job
4. vLLM exits cleanly (code 0), batch script gets SIGTERM (exit 0:15)
5. All SLURM steps cancelled simultaneously

The job appears healthy on the compute side — "vLLM is ready", API responding — but gets killed from the client side.

### Root Cause

`src/commands/start.ts` has a tunnel exit handler:
```typescript
tunnel.on('exit', (code) => {
  if (!shuttingDown)
    shutdown(`SSH tunnel exited unexpectedly (code ${code})`, 1);
});
```

When SSH fails to bind (port already in use with `ExitOnForwardFailure=yes`), it exits non-zero, which triggers the shutdown cascade.

### Fix

Pre-flight port check added before job submission in `src/commands/start.ts`:
1. Check if local port is in use via `lsof`
2. If occupied, show clear error with process name, PID, and suggested `kill` command
3. Fail fast — no SLURM job submitted

### Action items

- [x] Add pre-flight port availability check
- [x] Add `isLocalPortInUse()` helper function
- [x] All 407 tests pass after the change

---

## ISSUE-006: vLLM emits empty `tool_calls: []` in every streaming chunk (API serializer bug)

**Status**: Open — awaiting upstream vLLM fix

**Severity**: High — crashes Pi's tool-call parser on every response

**Affected**: DeepSeek-V4-Pro via vLLM 0.22.1 OpenAI-compatible API (`deepseek_v4` tokenizer mode)

### Symptom

Every streaming chunk from vLLM includes `"tool_calls": []` even when no tool calls are present:

```json
data: {"choices":[{"delta":{"content":"Hello","tool_calls":[]}}]}
data: {"choices":[{"delta":{"content":"!","tool_calls":[]}}]}
data: {"choices":[{"delta":{"content":" How","tool_calls":[]}}]}
```

Pi's streaming parser sees `tool_calls` in the delta, tries to access `tool_calls[0].function.input`, and crashes:
```
Cannot read properties of undefined (reading 'input')
```

This happens with **every response**, including simple text replies where no tools were requested. The empty `tool_calls` field makes clients believe a tool call is present.

### Root Cause

vLLM's `DeltaMessage` serializer in the OpenAI-compatible API includes `tool_calls: []` (empty array) instead of omitting the field when there are no tool calls. This is a **vLLM 0.22.1 bug** in the response serialization protocol.

Expected behaviour: omit `tool_calls` entirely when the array is empty (Pydantic `model_dump(exclude_none=True)` should handle this, but empty arrays are not `None`).

### Impact on Pi

Pi's `openai-completions` API handler expects:
- No `tool_calls` field → normal text response
- `tool_calls` field with entries → tool call in progress
- `tool_calls: []` → **undefined** — the client enters the tool-call path but the first element is undefined

This makes DeepSeek-V4-Pro completely unusable through Pi when running on raw vLLM. OpenRouter normalises the response format and strips the empty array, so it works through that proxy.

### Upstream tracking

| Item | Link | Status |
|---|---|---|
| Bug report | [vllm-project/vllm#44104](https://github.com/vllm-project/vllm/issues/44104) | Open (2 comments) |
| Related PR | [vllm-project/vllm#43155](https://github.com/vllm-project/vllm/pull/43155) | Open — Kimi forced tools |

### Reproduction (no GPU needed)

Minimal reproducer from upstream issue:
```python
from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage
from vllm.entrypoints.openai.engine.protocol import DeltaMessage

# Buggy: emits tool_calls: [] for plain text
d = DeltaMessage(content="hello", tool_calls=[])
print(d.model_dump(exclude_none=True))
# Actual:   {'content': 'hello', 'tool_calls': []}
# Expected: {'content': 'hello'}
```

### Workaround

**None at the vLLM level.** Must either:
1. Use OpenRouter or another proxy that normalises the response
2. Patch vLLM's `DeltaMessage` serialization to exclude empty `tool_calls` arrays
3. Wait for upstream fix

### Action items

- [ ] Document our Pi-specific crash as a data point on upstream issue #44104
- [ ] Test whether patching DeltaMessage locally resolves the Pi crash
- [ ] Update `min-vllm-version` when upstream fix lands

---

## ISSUE-003: Ray log archival captures thousands of irrelevant files on failure

**Status**: Known, low priority — not broken, just inconvenient

**Severity**: Low — affects post-failure diagnostics only

**Related**: SLURM template `persist_ray_logs` function

### Symptom

When a multi-node vLLM job fails, `persist_ray_logs` copies the entire Ray session log directory from each node to `ray-logs/` in the job working directory. This produces ~3,700+ files across 4 nodes (937 per node), most of which are empty or contain Ray internal tracing irrelevant to debugging.

### File breakdown (per node, head node example)

| Category | Count | Content |
|---|---|---|
| `worker-*.out` (empty) | ~292 | Empty stdout |
| `python-core-worker-*.log` | 294 | Ray internal tracing |
| `dashboard_*.err/log/out` | 24 | Dashboard internals |
| `runtime_env_*` | 3 | Env setup logs |
| **Signal** | | |
| `gcs_server.err/out` | 2 | GCS cluster-level issues (121 KB) |
| `raylet.err/out` | 2 | Raylet daemon issues (129 KB) |
| `monitor.err/log` | 3 | Actor lifecycle (33 KB) |
| `log_monitor.log` | 1 | Log management (93 KB) |
| `debug_state.txt` | 1 | Cluster state dump (23 KB) |
| `events` | 1 | Ray event log |
| Non-empty `worker-*.err` | ~few | Actual worker errors |

~95% of the 937 files are noise. The actual signal is in ~15 files.

### Proposed fix

Replace the current `cp -a` of the entire Ray logs directory with a selective copy:

1. **Always capture** (one per node): `gcs_server.*`, `raylet.*`, `monitor.*`, `log_monitor.*`, `debug_state.txt`, `events`
2. **Conditionally capture**: only `worker-*.err` files that are non-empty (actual errors)
3. **Drop**: all `worker-*.out`, `python-core-worker-*.log`, `dashboard_*`, `runtime_env_*`

This reduces ~937 files per node → ~15 files per node, making the diagnostics much easier to navigate.

### Action items

- [ ] Update `persist_ray_logs` in SLURM templates to selectively copy only useful files
- [ ] Update `collect_exit_diagnostics` documentation

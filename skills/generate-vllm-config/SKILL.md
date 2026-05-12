---
name: generate-vllm-config
description: 'Use when asked to generate, create, or write a vLLM config, vllm.yaml, or configuration file for a HuggingFace model to run on Isambard AI HPC. Triggers on phrases like "generate a config for <model>", "create a vllm.yaml for <model>", "what config do I need for <model>", "set up config for llama/qwen/mistral/deepseek". Do NOT use for general vLLM questions unrelated to config file generation.'
license: MIT
metadata:
  author: Rob Challen
  version: 0.1
---

# Generate vLLM Config for Isambard AI

Generates a ready-to-use `vllm.yaml` config file for running a HuggingFace model on Isambard AI HPC using `ivllm start`. It fetches the model card to determine architecture and parameter count, calculates memory requirements against Isambard AI's GH200 120GB nodes (~96 GiB usable HBM3e per GPU), selects appropriate parallelism, and writes the YAML file.

## When to Use This Skill

- User asks to generate or create a vLLM config or `vllm.yaml` for a named model
- User asks "what config do I need to run \<model\> on Isambard?"
- User is about to run `ivllm start` and needs a config file
- User provides a HuggingFace model ID and wants to know how many GPUs are needed

Do NOT use for general vLLM help, `ivllm` troubleshooting, or non-config questions.

## Prerequisites

- **Model ID**: A HuggingFace model ID (e.g. `Qwen/Qwen2.5-72B-Instruct`). Ask the user if not provided.
- **Output filename**: Ask the user — default is `vllm.yaml` in the current directory.
- **Web access**: Required to fetch the model card from `https://huggingface.co/<model-id>`.
- **Write permission**: To save the YAML file.

## Steps

### 1. Confirm inputs

If the model ID was not provided as an argument, ask the user:
- Which HuggingFace model they want to use
- What filename to write to (default: `vllm.yaml`)

### 2. Fetch the model card

Fetch `https://huggingface.co/<model-id>` and extract:

| Field | Where to find it | Example |
|-------|-----------------|---------|
| Total parameter count | Model Overview table or description | "72.7B parameters" |
| Architecture type | Description — is it dense or MoE? | "Mixture of Experts" |
| Active parameters (MoE only) | Model Overview table | "3.3B activated" |
| Native context length | Model card — "Context Length" | "128,000 tokens" |
| Recommended dtype | Model card or vLLM quickstart snippet | "bfloat16" |
| Reasoning/thinking support | Any mention of `--reasoning-parser`, `enable_thinking`, or chain-of-thought | Yes/No |

If the model card is not accessible or the parameter count cannot be determined, ask the user to provide it.

### 3. Look up the official vLLM recipe (if available)

vLLM recipes can be queried using the huggingface model-id and hardware. Isambard AI's GH200 fals between H100 and H200 in terms of specifications:

e.g.:

`https://recipes.vllm.ai/<model-id>?hardware=h100`
`https://recipes.vllm.ai/<model-id>?hardware=h200`

Isambard AIs GH200 is a Grace CPU + H100 GPU and falls between H100 and H200 in terms of specifications (closer to H100).
Each node has 4 NVIDIA GH200 Grace Hopper Superchips with NVIDIA NVLink-C2C interconnect. Each node has 460 GB of usable CPU memory (115 GB is usable for each CPU), and 384 GB of GPU memory. In total, there is 844 GB of CPU + GPU memory per node.

If a recipe exists:
- Note the minimum vLLM version that supports the model
- Note the recommended `tensor-parallel-size`, quantization, and any special flags
- Note whether `--enable-expert-parallel` is recommended (always the case for large MoE models)
- Note the `reasoning-parser` name if applicable
- Find out what specific options are recommended and why.
- Adapt the recipe's recommendations to Isambard AI's 4 × GH200 120GB topology (~96 GiB usable per GPU)

The recipes are written for various hardware; translate GPU counts to Isambard AI node counts using: `nodes = ceil(recipe_gpus / 4)`.

### 4. Establish vLLM minimum version

- **min-vllm-version**: Set this to the minimum vLLM version required to run the model. `ivllm start` checks this against the installed vLLM version and fails early if not satisfied. This key is **stripped before the config is passed to `vllm serve`** (vLLM errors on unknown keys). Use the earliest vLLM version in which the model and any required parsers/features were added. If unsure, check the vLLM changelog or recipes page, or set to `"0.19.1"` as a conservative baseline (the current default install version).

vLLM CLI options and config keys vary between releases. Always consult the documentation for the specific version being deployed:

- **All versions index**: https://app.readthedocs.org/projects/vllm/
- **Versioned serve CLI docs**: `https://docs.vllm.ai/en/v<version>/cli/serve/`
  e.g. https://docs.vllm.ai/en/v0.19.1/cli/serve/

When checking whether a feature, parser, or config key exists, verify it in the docs for the version specified by `min-vllm-version` in the config (or the user's configured `vllm-version`). Avoid referencing `/en/latest/` when the deployed version is fixed.

### 5. Calculate memory and parallelism

Use the rules in [references/vllm-config-guide.md](references/vllm-config-guide.md):

```
weights_GB ≈ total_params_B × 2          (bfloat16, both dense and MoE)
usable_per_gpu = 86.4 GB                 (96 GB × 0.90 utilization; GH200 reports ~95.6 GiB usable)
needed_gpus = ceil(weights_GB / usable_per_gpu)
```

**GH200 unified memory + CPU offload (`--cpu-offload-gb`):**

The GH200's NVLink-C2C (900 GB/s coherent interconnect) creates a unified memory address space between CPU (460 GB) and GPU (96 GB). The `--cpu-offload-gb` flag reserves CPU memory space per GPU for model weights, accessed via UVA zero-copy during forward passes. This is effectively a virtual way to increase GPU memory — e.g., 24 GB GPU + 10 GB offload = 34 GB virtual capacity.

Use `--cpu-offload-gb` if:
- The model's weight memory exceeds what fits on available GPUs even at fp8
- The user explicitly requests it or asks about memory optimization for very large models
- It requires vLLM ≥ 0.6.0 (feature availability)

**Example:**
```yaml
model: meta-llama/Llama-3.1-70B
tensor-parallel-size: 1               # 70B @ bf16 ≈ 140 GB; exceeds single 96GB GPU
# On single GPU with offload:
cpu-offload-gb: 50                  # reserves 50 GB CPU memory (passed as CLI arg)
gpu-memory-utilization: 0.90        # uses ~86 GB GPU (140-50 = 90 GB, close to limit)
```

When reporting to the user, include:
- "CPU offload: cpu-offload-gb <N>"
- "Requires fast CPU-GPU interconnect (NVLink-C2C on GH200)"
- "Minimum vLLM version: 0.6.0+"

**Single node** (needed_gpus ≤ 4):
- **Default to `tensor-parallel-size: 4` (full node) for any non-trivial model.**
  There is no penalty to using all 4 GPUs on a single node — it gives 4× more KV cache space (directly enabling longer contexts and higher batch throughput), leverages NVLink-C2C for near-linear TP scaling, and keeps the job within a single node (simpler, lower latency). Only drop below tp=4 if:
  - The model is very small (< ~7B) and the user explicitly asks for a minimal footprint, **or**
  - The model's number of attention heads is not divisible by 4 (rare; check the model card)
- `pipeline-parallel-size` = 1 (omit from config)

**Multi-node** (needed_gpus > 4):
- `tensor-parallel-size` = 4
- `pipeline-parallel-size` = ceil(needed_gpus / 4)
- ⚠️ **Warn the user**: multi-node jobs require more resources and inter-node communication adds latency. `ivllm start` supports multi-node via Ray automatically, but confirm the user needs multi-node before proceeding.

### 6. Choose max-model-len

- **Default to the model's full native context length.** With tp=4 on 4 × GH200 120GB (~345 GB usable VRAM combined), KV cache headroom is large even for very long contexts — do not reduce the context pre-emptively.
- Only reduce `max-model-len` if an explicit OOM analysis shows the KV cache at native context would exhaust available memory after weights are loaded. Calculate:
  ```
  kv_per_token ≈ num_kv_heads × head_dim × 2 bytes × num_layers  (standard dense attention)
  total_kv_GB  = kv_per_token × max_tokens / 1e9
  available_for_kv = (usable_per_gpu × tensor_parallel_size) − weights_GB
  ```
- If the native context does exceed available KV budget, reduce to the largest power-of-two that fits, and note the native context in a YAML comment.
- Exception: hybrid architectures (e.g. Qwen3.5-35B-A3B with Gated DeltaNet layers) have a tiny KV footprint — keep the full context.

### 7. Check for special options

- **Reasoning models** (Qwen3, DeepSeek-R1, QwQ, etc.): add the `reasoning-parser`. Use `qwen3` for Qwen3 series; `deepseek_r1` for DeepSeek-R1/V3 and most others. Check the model card vLLM quickstart snippet for the exact name.
- **MoE models with `tensor-parallel-size >= 2`**: add `enable-expert-parallel: true`. The official vLLM recipes (DeepSeek-R1, Qwen3.5) consistently recommend this — it uses expert parallelism for the MoE layers (all-to-all comms, more efficient) while dense layers remain tensor-parallelized. No benefit when tp=1.
- **FP8 quantization**: GH200/H100 (Hopper) has native FP8 tensor cores. If the model is memory-constrained or throughput is important, suggest `quantization: fp8`. This halves weight memory (`params_B × 1 GB` vs `× 2 GB`). Check if a pre-quantized `-FP8` variant exists on HuggingFace — prefer it over runtime quantization.
- **Tool calling**: Always include `enable-auto-tool-choice: true` and the matching `tool-call-parser` unless the model is known not to support function calling (e.g. base/pretrain checkpoints, pure reasoning models without tool support). The parser is required — without it, tool call responses come back as raw text rather than structured `tool_calls` objects. See the tool-call parser table in `references/vllm-config-guide.md`.
- **Prefix caching**: Recommend `enable-prefix-caching: true` for agent/chatbot use cases with repeated system prompts. Low cost, high benefit.

### 7. Write the YAML file

Write the config to the requested filename. Include comments to explain the key choices:

```yaml
# vLLM config for <model-id> on Isambard AI
# Generated by ivllm generate-vllm-config
#
# Hardware: 4 × GH200 120GB per node (Isambard AI Phase 1/2) — ~96 GiB usable HBM3e per GPU
# Model: <X>B parameters, ~<Y> GB at bfloat16
# Parallelism: <explanation>

model: <model-id>
tensor-parallel-size: 4
# pipeline-parallel-size: <M>   # for multi-node: each pipeline stage = 1 node (tp=4)
max-model-len: <native_context_length>
gpu-memory-utilization: 0.90
dtype: bfloat16
enable-auto-tool-choice: true
tool-call-parser: <parser>      # see vllm-config-guide.md for parser names
enable-prefix-caching: true
min-vllm-version: "<version>"
# min-vllm-version: minimum vLLM version to run this model; ivllm checks before
# submitting. Stripped before passing to vllm serve (vLLM errors on unknown keys).
```

Only include keys that are non-default or important — keep the config minimal and readable.

### 8. Report to the user

After writing the file, tell the user:
- The filename written
- The model size and memory calculation
- The parallelism chosen and why (tp=4 is default; note if reduced and why)
- How to use it: `ivllm start <job-name> --config <filename>`
- Any warnings (multi-node required, context was reduced from native, etc.)

## Guidance

### Memory calculation principles

The rule of thumb `params_B × 2 GB` for bfloat16 is conservative (it's exact for weights; the actual overhead includes activations and KV cache, handled by `gpu-memory-utilization`). When in doubt, round up.

For MoE models, all expert weights must reside in GPU memory simultaneously even though only a few activate per forward pass — use total parameter count, not active parameter count, for the memory calculation.

### CPU offload (`--cpu-offload-gb`)

**Key facts from NVIDIA's GH200 unified memory research:**
- NVLink-C2C connects CPU and GPU at 900 GB/s — 7× PCIe Gen 5 bandwidth, memory-coherent
- Creates a single unified memory address space: CPU (120 GB) + GPU (96 GB) share one page table and 4x superchips per node
- `--cpu-offload-gb` uses UVA (Unified Virtual Addressing) for zero-copy access — weights are loaded from CPU memory to GPU memory on-the-fly during each forward pass
- The offload parameter is "virtual GPU memory" — e.g., 24 GB GPU + 10 GB offload = 34 GB virtual capacity
- Requires fast CPU-GPU interconnect (NVLink-C2C on GH200 makes this practical)

**Configuration options (from vLLM OffloadConfig):**

| Option | Description | Default | Notes |
|--------|-------------|---------|-------|
| `--cpu-offload-gb` | CPU memory (GiB) per GPU for weights | 0 | Must use UVA backend |
| `--offload-backend` | "auto", "prefetch", or "uva" | auto | "uva" for cpu-offload-gb; "prefetch" for async |
| `--offload-group-size` | Group N layers, offload last N of each group | 0 | Uses async prefetch, not UVA |
| `--offload-num-in-group` | Layers to offload per group | 1 | Must be ≤ offload-group-size |

**When to suggest CPU offload:**
- Model exceeds single-GPU memory (e.g., Llama-3-70B at 140 GB vs 96 GB GPU)
- Single-node job desired but weights don't fit
- User asks about optimization for large models

**When NOT to suggest:**
- Multi-node is appropriate (better throughput)
- fp8 quantization solves the problem

**Note:** `--cpu-offload-gb` maps to `cpu-offload-gb` in YAML config — all vLLM CLI arguments work as YAML keys (with `--` stripped).

### Parallelism choices

**The default is tp=4 (full node).** On Isambard AI there is no queuing penalty for using all 4 GH200s on a single node — the purpose of the system is large-scale inference. Using the full node gives:
- 4× more aggregate KV cache memory, enabling longer contexts and larger batches
- Higher throughput via NVLink-C2C-connected tensor parallelism (near-linear scaling)
- Full use of the allocated node allocation

Reduce below tp=4 only when:
- **tp=1**: model is very small (< ~7B) and the user explicitly wants a minimal single-GPU footprint
- **tp=2**: attention head count is not divisible by 4 but is divisible by 2 (rare with modern models)

Multi-node (pp>1) adds inter-node communication latency and complexity — only use it when the model's weights genuinely won't fit on a single 4-GPU node.

Valid `tensor-parallel-size` values must divide the model's number of attention heads evenly. 1, 2, and 4 work for virtually all modern models.

### Context length

**Default to the model's full native context length.** With tp=4 across 4 × GH200 120GB (~345 GB usable), there is substantial KV cache headroom. The point of running on Isambard AI is to go big — large contexts, high throughput.

Only reduce `max-model-len` if an explicit calculation shows the KV cache at native context would exhaust remaining memory after weights are loaded. If reduction is needed, note the native context in a comment and reduce to the next power-of-two that fits. Never silently cap at 32K for a model that supports 128K+ unless forced by memory.

### Multi-node warning

## Multi-node

`ivllm start` supports multi-node SLURM jobs via Ray. When `pipeline-parallel-size > 1`, it automatically sets `--nodes=N` and bootstraps a Ray cluster across those nodes before starting vLLM with `--distributed-executor-backend ray`. No manual SLURM setup required.

## Validation

Before writing the file, verify:
- [ ] `tensor-parallel-size × pipeline-parallel-size` = total GPUs needed
- [ ] `tensor-parallel-size` is 1, 2, or 4
- [ ] `max-model-len` is a positive integer ≤ native context length
- [ ] `model` field exactly matches the HuggingFace model ID (case-sensitive)
- [ ] If multi-node: user has been warned about resource requirements (multi-node is supported by `ivllm`)
- [ ] `min-vllm-version` is set to the earliest vLLM version that supports this model

## Troubleshooting

| Problem | Solution |
|---------|---------|
| Model card not accessible | Ask user for parameter count and architecture manually |
| Parameter count ambiguous | Use total parameters (not non-embedding, not active) for memory calculation |
| Model fits on paper but OOM in practice | Reduce `max-model-len` by half; or suggest `quantization: fp8` |
| Context reduced from native | Always note native context in a comment; reduce to next power-of-two that fits, not to an arbitrary 32K |
| Reasoning parser name unknown | Check the model card vLLM quickstart snippet — it usually names the parser explicitly; or check the recipes page |
| MoE model shows poor throughput | Add `enable-expert-parallel: true` with `tensor-parallel-size >= 2` |
| Memory borderline | Try `quantization: fp8` — halves weight memory with minimal accuracy loss on Hopper (GH200/H100) |
| `ivllm start` fails with "version too low" | The installed vLLM (`ivllm config --vllm-version`) is below `min-vllm-version`; run `ivllm setup` with a newer version or update the config |

## References

- [Isambard AI hardware specs](references/isambard-specs.md)
- [vLLM config options and memory guide](references/vllm-config-guide.md)
- [vLLM model-specific recipes](https://docs.vllm.ai/projects/recipes/en/latest/)
- [Isambard AI specs online](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-2)


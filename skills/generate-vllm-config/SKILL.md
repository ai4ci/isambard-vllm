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

### 2. Look up the official vLLM recipe (if available)

Before calculating from scratch, check `https://docs.vllm.ai/projects/recipes/en/latest/` for a model-specific recipe. The recipes page lists community-maintained guides for common models. If a recipe exists:
- Note the recommended `tensor-parallel-size`, quantization, and any special flags
- Note whether `--enable-expert-parallel` is recommended (always the case for large MoE models)
- Note the `reasoning-parser` name if applicable
- Adapt the recipe's recommendations to Isambard AI's 4 × GH200 120GB topology (~96 GiB usable per GPU)

The recipes are written for various hardware; translate GPU counts to Isambard AI node counts using: `nodes = ceil(recipe_gpus / 4)`.

### 3. Fetch the model card

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

### 4. Calculate memory and parallelism

Use the rules in [references/vllm-config-guide.md](references/vllm-config-guide.md):

```
weights_GB ≈ total_params_B × 2          (bfloat16, both dense and MoE)
usable_per_gpu = 86.4 GB                 (96 GB × 0.90 utilization; GH200 reports ~95.6 GiB usable)
needed_gpus = ceil(weights_GB / usable_per_gpu)
```

**Single node** (needed_gpus ≤ 4):
- `tensor-parallel-size` = needed_gpus, rounded up to 1, 2, or 4
- `pipeline-parallel-size` = 1 (omit from config)

**Multi-node** (needed_gpus > 4):
- `tensor-parallel-size` = 4
- `pipeline-parallel-size` = ceil(needed_gpus / 4)
- ⚠️ **Warn the user**: multi-node jobs require more GPUs and resources. Confirm they need multi-node before proceeding. `ivllm start` supports multi-node via Ray — it will automatically request the required number of nodes.

### 5. Choose max-model-len

- Start with the model's native context length
- If the model supports >64K tokens natively, suggest 32768 as a practical default (KV cache for very long contexts consumes significant GPU memory)
- Note the native context length in a YAML comment so the user knows what they are giving up

### 6. Check for special options

- **Reasoning models** (Qwen3, DeepSeek-R1, QwQ, etc.): add `enable-reasoning: true` and the `reasoning-parser`. Use `qwen3` for Qwen3 series; `deepseek_r1` for DeepSeek-R1/V3 and most others. Check the model card vLLM quickstart snippet for the exact name.
- **MoE models with `tensor-parallel-size >= 2`**: add `enable-expert-parallel: true`. The official vLLM recipes (DeepSeek-R1, Qwen3.5) consistently recommend this — it uses expert parallelism for the MoE layers (all-to-all comms, more efficient) while dense layers remain tensor-parallelized. No benefit when tp=1.
- **FP8 quantization**: GH200/H100 (Hopper) has native FP8 tensor cores. If the model is memory-constrained or throughput is important, suggest `quantization: fp8`. This halves weight memory (`params_B × 1 GB` vs `× 2 GB`). Check if a pre-quantized `-FP8` variant exists on HuggingFace — prefer it over runtime quantization.
- **Tool calling**: Always include `enable-auto-tool-choice: true` and the matching `tool-call-parser` unless the model is known not to support function calling (e.g. base/pretrain checkpoints, pure reasoning models without tool support). The parser is required — without it, tool call responses come back as raw text rather than structured `tool_calls` objects. See the tool-call parser table in `references/vllm-config-guide.md`.
- **Prefix caching**: Recommend `enable-prefix-caching: true` for agent/chatbot use cases with repeated system prompts. Low cost, high benefit.
- **min-vllm-version**: Set this to the minimum vLLM version required to run the model. `ivllm start` checks this against the installed vLLM version and fails early if not satisfied. This key is **stripped before the config is passed to `vllm serve`** (vLLM errors on unknown keys). Use the earliest vLLM version in which the model and any required parsers/features were added. If unsure, check the vLLM changelog or recipes page, or set to `"0.9.1"` as a conservative baseline.

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
tensor-parallel-size: <N>
# pipeline-parallel-size: <M>   # for multi-node: each pipeline stage = 1 node (tp=4)
max-model-len: <context_length>
# Native context: <native_context> — reduced to save KV cache memory
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
- The parallelism chosen and why
- How to use it: `ivllm start <job-name> --config <filename>`
- Any warnings (multi-node, context reduction, etc.)

## Guidance

### Memory calculation principles

The rule of thumb `params_B × 2 GB` for bfloat16 is conservative (it's exact for weights; the actual overhead includes activations and KV cache, handled by `gpu-memory-utilization`). When in doubt, round up.

For MoE models, all expert weights must reside in GPU memory simultaneously even though only a few activate per forward pass — use total parameter count, not active parameter count, for the memory calculation.

### Parallelism choices

- **tp=1**: preferred when the model fits on a single GPU — no communication overhead
- **tp=2**: when the model needs 2 GPUs; NVLink-C2C gives near-linear scaling on Isambard AI
- **tp=4**: uses the whole node; ideal for the largest single-node models
- **pp>1**: only for multi-node; each pipeline stage is one full node (tp=4); latency increases with depth

Valid `tensor-parallel-size` values depend on the number of attention heads — it must divide the head count evenly. 1, 2, and 4 work for virtually all modern models.

### Context length trade-offs

The KV cache for a 128K context can be larger than the model weights for small models. If the model card says "supports 128K" but the use case is interactive chat, 32K is usually sufficient and much more memory efficient. Always leave the native context length in a comment so the user can restore it if needed.

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
| User insists on a context length that won't fit | Explain the trade-off and suggest the maximum feasible context given the GPU budget |
| Reasoning parser name unknown | Check the model card vLLM quickstart snippet — it usually names the parser explicitly; or check the recipes page |
| MoE model shows poor throughput | Add `enable-expert-parallel: true` with `tensor-parallel-size >= 2` |
| Memory borderline | Try `quantization: fp8` — halves weight memory with minimal accuracy loss on Hopper (GH200/H100) |
| `ivllm start` fails with "version too low" | The installed vLLM (`ivllm config --vllm-version`) is below `min-vllm-version`; run `ivllm setup` with a newer version or update the config |

## References

- [Isambard AI hardware specs](references/isambard-specs.md)
- [vLLM config options and memory guide](references/vllm-config-guide.md)
- [vLLM model-specific recipes](https://docs.vllm.ai/projects/recipes/en/latest/)
- [vLLM serve help text](references/help.txt)
- [Isambard AI specs online](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-2)

### vLLM versioned documentation

vLLM CLI options and config keys vary between releases. Always consult the documentation for the specific version being deployed:

- **All versions index**: https://app.readthedocs.org/projects/vllm/
- **Versioned serve CLI docs**: `https://docs.vllm.ai/en/v<version>/cli/serve/`  
  e.g. https://docs.vllm.ai/en/v0.19.1/cli/serve/

When checking whether a feature, parser, or config key exists, verify it in the docs for the version specified by `min-vllm-version` in the config (or the user's configured `vllm-version`). Avoid referencing `/en/latest/` when the deployed version is fixed.

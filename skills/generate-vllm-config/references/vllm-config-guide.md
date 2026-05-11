# vLLM Config Guide for Isambard AI

## Config file format

vLLM accepts a YAML config file (passed via `--config <file>`) containing any CLI option in kebab-case:

```yaml
model: Qwen/Qwen2.5-72B-Instruct
tensor-parallel-size: 2
max-model-len: 32768
gpu-memory-utilization: 0.90
dtype: bfloat16
```

## Tool calling

Tool calling requires two YAML keys (in addition to the model's chat template, which is bundled with the tokenizer):

```yaml
enable-auto-tool-choice: true
tool-call-parser: <parser>
```

Without `tool-call-parser`, the model's raw text response is returned — structured `tool_calls` objects in the API response won't be populated.

### Tool-call parser by model family

| Model family | `tool-call-parser` value |
|-------------|--------------------------|
| Qwen3 series (Qwen3-*, Qwen3-MoE) | `qwen3_coder` |
| Qwen2.5 Instruct series | `hermes` |
| Llama 3.x (3.1, 3.2, 3.3) | `llama3_json` |
| Llama 4 series | `llama4_json` |
| DeepSeek-V3 | `deepseek_v3` |
| DeepSeek-V3-0324 / newer | `deepseek_v31` |
| Mistral / Mixtral | `mistral` |
| InternLM | `internlm` |
| IBM Granite | `granite` |
| Generic OpenAI-compatible | `openai` |

If the model family isn't listed, check the model card for a vLLM quickstart — it usually names the parser. If still unsure, try `hermes` (widely supported) or check `--tool-call-parser` in the vLLM help for the full list.

**When to omit tool calling:** Base/pretrain checkpoints and pure reasoning models (e.g. DeepSeek-R1 without tool fine-tuning) don't support tool calls. If the model card says "Instruct" or "Chat", assume tool calling is intended.



| YAML key | Description | Typical value |
|----------|-------------|---------------|
| `model` | HuggingFace model ID | Required |
| `tensor-parallel-size` | GPUs used within a single node | 1–4 |
| `pipeline-parallel-size` | Nodes used (tp × pp = total GPUs) | 1 (single node) |
| `max-model-len` | Max context length in tokens | Reduce if memory constrained |
| `gpu-memory-utilization` | Fraction of GPU memory for weights + KV cache | 0.90 |
| `dtype` | Weight dtype | `bfloat16` (H100 native) |

## Memory estimation

### Dense models (Llama, Qwen dense, etc.)

Weight memory (GB) ≈ `parameters_B × 2` for bfloat16 (2 bytes per param)

Examples:
| Model | Params | Weight GB (bf16) | Minimum GPUs (96 GB each) |
|-------|--------|-------------------|--------------------------|
| Qwen2.5-7B | 7B | ~14 GB | 1 |
| Llama-3.1-8B | 8B | ~16 GB | 1 |
| Qwen2.5-72B | 72B | ~144 GB | 2 |
| Llama-3.1-70B | 70B | ~140 GB | 2 |
| Llama-3.1-405B | 405B | ~810 GB | 9 (multi-node) |

### MoE models (Qwen3-MoE, Mixtral, DeepSeek, etc.)

All experts must be loaded into memory even though only a subset activate per token:
Weight memory (GB) ≈ `total_parameters_B × 2` for bfloat16

Examples:
| Model | Total params | Active params | Weight GB (bf16) | Minimum GPUs |
|-------|-------------|---------------|-------------------|--------------|
| Qwen3-30B-A3B | 30B | 3.3B | ~60 GB | 1 |
| Mixtral-8x7B | 46B | 13B | ~92 GB | 1–2 |
| DeepSeek-R1 | 671B | ~37B | ~1340 GB | multi-node |

### KV cache overhead

At `gpu-memory-utilization: 0.90`, roughly 60–75% of usable memory is weights, 25–40% is KV cache. 
If `max-model-len` is large (>32K), KV cache can dominate — reduce `max-model-len` to reclaim.

## Parallelism decision tree

```
weights_GB = params_B × 2   (bf16)  OR  params_B × 1  (fp8/quantized)
usable_per_gpu = 96 × 0.90 = 86.4 GB

needed_gpus = ceil(weights_GB / usable_per_gpu)

if needed_gpus <= 4:
    tensor-parallel-size = needed_gpus  (round up to 1, 2, or 4)
    pipeline-parallel-size = 1
    → single node job
else:
    tensor-parallel-size = 4
    pipeline-parallel-size = ceil(needed_gpus / 4)
    → multi-node job  ⚠️  WARN USER (not yet supported by ivllm)
```

Valid `tensor-parallel-size` values: 1, 2, 4 (must divide number of attention heads evenly).

## GH200 unified memory + CPU offloading

The GH200 (Grace Hopper Superchip) connects CPU and GPU via NVLink-C2C — a 900 GB/s memory-coherent interconnect. This creates a single unified memory address space shared by both CPU (460 GB) and GPU (96 GB HBM3).

Two mechanisms let models exceed per-GPU VRAM:

**Unified virtual addressing (automatic):** The GPU transparently accesses CPU memory when page faults occur. No special config needed — it just works. The GPU memory shown by `gpu-memory-utilization` governs the GPU-side split; overflow goes to CPU memory transparently.

**Explicit CPU offload (`--cpu-offload-gb`):** Reserves a fixed space in CPU memory per GPU for model weights, accessed via UVA zero-copy on-the-fly during each forward pass. Effectively increases the GPU's capacity (e.g. 24 GB GPU + 10 GB offload = 34 GB virtual).

For Isambard's GH200, the NVLink-C2C interconnect (900 GB/s vs ~24 GB/s for PCIe) makes CPU offload practical with minimal latency penalty. Only consider this if the model genuinely doesn't fit on a full 4-GPU node at fp8.

To enable CPU offloading in vLLM config:
```yaml
# cpu-offload-gb is passed via --cpu-offload-gb <n> on the vllm serve command
# It is NOT a YAML key — it's a CLI argument passed when vllm serve starts
# ivllm handles this internally if the config has:
_min-vllm-version: "0.6.0"   # offload was added in v0.6.0+
```

## Quantization

If a model barely fits but is too large at bf16, suggest `fp8` quantization (halves weight memory):
```yaml
quantization: fp8
```
H100 (and GH200) has native fp8 tensor cores — minimal accuracy loss, ~2× throughput improvement.

## MoE-specific: expert parallelism

For MoE models (Qwen3-MoE, Mixtral, DeepSeek, etc.) with `tensor-parallel-size >= 2`, use `enable-expert-parallel: true`.

vLLM uses expert parallelism **for the MoE expert layers** instead of tensor parallelism. Dense layers (attention, etc.) are still tensor-parallelized. Expert parallelism assigns different experts to different GPUs (all-to-all comms) rather than sharding each expert across GPUs (all-reduce comms) — far more efficient for MoE workloads.

The vLLM official recipes (DeepSeek-R1, Qwen3.5) consistently recommend this pattern: `tp=N --enable-expert-parallel`.

```yaml
# MoE model example (Qwen3-30B-A3B on 1 GPU)
model: Qwen/Qwen3-30B-A3B
tensor-parallel-size: 1
# enable-expert-parallel not needed for tp=1 (only 1 GPU)
```

```yaml
# Large MoE on full node (tp=4 + expert parallel)
model: deepseek-ai/DeepSeek-R1
tensor-parallel-size: 4
enable-expert-parallel: true
quantization: fp8
max-model-len: 32768
gpu-memory-utilization: 0.90
dtype: bfloat16
reasoning-parser: deepseek_r1
enable-reasoning: true
```

## Reasoning models (Qwen3, DeepSeek-R1, etc.)

Some models support a "thinking" mode that generates chain-of-thought reasoning.
If the model card mentions reasoning/thinking, add:
```yaml
enable-reasoning: true
reasoning-parser: deepseek_r1  # DeepSeek-R1, DeepSeek-V3, and most reasoning models
# reasoning-parser: qwen3      # Qwen3 series specifically
```

**Parser names by model family:**
| Model family | reasoning-parser value |
|-------------|------------------------|
| Qwen3 series | `qwen3` |
| DeepSeek-R1, DeepSeek-V3 | `deepseek_r1` |
| QwQ | `deepseek_r1` |
| Others | Check vLLM recipes page or model card |

## FP8 quantization — Hopper-optimised

H100 (Hopper architecture, as on Isambard AI) has native FP8 tensor cores. The vLLM Llama recipe explicitly notes: *"For Hopper, FP8 offers the best performance for most workloads."*

FP8 halves weight memory vs bfloat16 (`params_B × 1 GB` instead of `× 2 GB`) with minimal accuracy loss and typically ~2× throughput improvement.

Use if: model is borderline for memory, or throughput is important:
```yaml
quantization: fp8
```

Many models have pre-quantized FP8 variants on HuggingFace (look for `-FP8` suffix) — these are preferable to runtime quantization.

## Prefix caching

For use cases with repeated system prompts or shared context (chatbots, agents), add:
```yaml
enable-prefix-caching: true
```
This is recommended in the official vLLM recipes for throughput-focused serving. It has no cost if prefixes don't repeat.

## max-model-len guidance

- Start with the model's native context length from the model card
- Reduce if you get OOM errors — halving context roughly halves KV cache memory
- For most use cases 32768 (32K) is a good starting point even if the model supports 128K+
- Very long contexts (>64K) are memory intensive; only use if needed

## Common complete configs

### Small model, single GPU
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

### Large dense model, 2 GPUs
```yaml
model: Qwen/Qwen2.5-72B-Instruct
tensor-parallel-size: 2
max-model-len: 32768
gpu-memory-utilization: 0.90
dtype: bfloat16
enable-auto-tool-choice: true
tool-call-parser: hermes
enable-prefix-caching: true
```

### Large dense model, FP8 (Hopper-optimised, ~2x throughput)
```yaml
model: meta-llama/Llama-3.3-70B-Instruct
tensor-parallel-size: 1       # FP8 halves memory; 70B fits on 1x96GB at FP8
max-model-len: 32768
gpu-memory-utilization: 0.90
quantization: fp8
enable-auto-tool-choice: true
tool-call-parser: llama3_json
enable-prefix-caching: true
```

### MoE reasoning model, single GPU
```yaml
model: Qwen/Qwen3-30B-A3B
tensor-parallel-size: 1
max-model-len: 32768
gpu-memory-utilization: 0.90
dtype: bfloat16
enable-reasoning: true
reasoning-parser: qwen3
enable-auto-tool-choice: true
tool-call-parser: qwen3_coder
enable-prefix-caching: true
```

### Large MoE, full node (4 GPUs) with expert parallelism + FP8
```yaml
# ~670B total params; FP8 ~670 GB → needs multi-node even at FP8
# Example shown for 4 GPUs — would need pipeline-parallel-size for real deployment
model: deepseek-ai/DeepSeek-R1
tensor-parallel-size: 4
# pipeline-parallel-size: 2   # uncomment for multi-node (not yet supported by ivllm)
max-model-len: 32768
gpu-memory-utilization: 0.90
quantization: fp8
enable-expert-parallel: true
enable-reasoning: true
reasoning-parser: deepseek_r1
# Note: DeepSeek-R1 is a reasoning model; tool calling requires fine-tuned variant
enable-prefix-caching: true
```

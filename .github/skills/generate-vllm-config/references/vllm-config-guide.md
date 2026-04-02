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

## Essential options

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

## Quantization

If a model barely fits but is too large at bf16, suggest `fp8` quantization (halves weight memory):
```yaml
quantization: fp8
```
H100 has native fp8 tensor cores — minimal accuracy loss, ~2× throughput improvement.

## MoE-specific: expert parallelism

For MoE models on a single node with tp < 4, consider `enable-expert-parallel: true`.
This distributes experts across GPUs rather than sharding each expert across GPUs — better utilization.
Only useful when `tensor-parallel-size < num_experts`.

## Reasoning models (Qwen3, DeepSeek-R1, etc.)

Some models support a "thinking" mode that generates chain-of-thought reasoning.
If the model card mentions reasoning/thinking, add:
```yaml
enable-reasoning: true
reasoning-parser: deepseek_r1  # or qwen3 for Qwen3 models
```

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
```

### Large dense model, 2 GPUs
```yaml
model: Qwen/Qwen2.5-72B-Instruct
tensor-parallel-size: 2
max-model-len: 32768
gpu-memory-utilization: 0.90
dtype: bfloat16
```

### MoE model, single GPU
```yaml
model: Qwen/Qwen3-30B-A3B
tensor-parallel-size: 1
max-model-len: 32768
gpu-memory-utilization: 0.90
dtype: bfloat16
enable-reasoning: true
reasoning-parser: deepseek_r1
```

### Full 4-GPU node
```yaml
model: meta-llama/Llama-3.1-405B-Instruct
tensor-parallel-size: 4
pipeline-parallel-size: 3   # 12 GPUs total, 3 nodes
max-model-len: 32768
gpu-memory-utilization: 0.90
dtype: bfloat16
```

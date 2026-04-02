# Isambard AI Hardware Specifications

Source: https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-2

## Compute Node (Phase 1 & Phase 2)

| Property | Value |
|----------|-------|
| GPUs per node | 4 × NVIDIA H100 Tensor Core GPU |
| GPU memory per GPU | 96 GB HBM3 |
| Total GPU memory per node | 384 GB |
| Intra-node GPU interconnect | NVIDIA NVLink-C2C (very high bandwidth, full BW between all 4 GPUs) |
| Inter-node network | Slingshot 11, 4 × 200 Gbps Cassini NICs per node |
| CPU architecture | aarch64 (ARM Grace) |
| CPU memory per node | 460 GB |

## Key implications for vLLM

- **Single-node GPU budget**: 384 GB total (`4 × 96 GB × 0.9 utilization = 345.6 GB usable`)
- **Tensor parallelism**: up to 4 on a single node (NVLink-C2C gives near-linear scaling)
- **Pipeline parallelism**: required for models exceeding single-node memory; each pipeline stage = 1 node (tp=4)
- **aarch64 architecture**: vLLM must be installed natively for ARM; no x86 containers. The `ivllm setup` command handles this.
- **dtype**: bfloat16 is optimal for H100 (native bf16 tensor cores); float16 also works

## Memory Budget per GPU

With `gpu-memory-utilization: 0.90`:
- Usable per GPU: `96 × 0.90 = 86.4 GB`
- Usable per node (4 GPUs): `345.6 GB`

## Phase 2 scale
- 1,320 compute nodes, 5,280 H100 GPUs total
- Same per-node specs as Phase 1

# Multi-Node vLLM Deployment on Isambard AI (GH200): Learnings & Best Practices

This document compiles the core technical findings, pitfalls, and solutions discovered during the deployment and tuning of multi-node Mixture of Experts (MoE) models (such as Qwen 3.5 397B and DeepSeek-V4-Pro) on **Isambard AI (NVIDIA GH200 Grace Hopper Superchip cluster)**. These learnings can be fed back to the Isambard HPC support team to help other users succeed with large-scale distributed inference.

---

## Executive Summary

Deploying extremely large models requires multi-node Tensor + Expert Parallel (TEP) or Pipeline Parallel configurations via **Ray** and **vLLM**. While single-node vLLM works relatively out-of-the-box, scaling to multiple nodes on a specialized Grace Hopper architecture connected via HPE Slingshot introduces unique challenges. 

Our key breakthroughs focused on **JIT compiler cache isolation on Lustre, bypassing CUDA IPC custom all-reduce over Slingshot, and ensuring robust environment propagation to Ray worker actors.**

---

## 1. JIT Compiler Cache Races on NFS vs. Lustre

### The Pitfall: NFS file-locking (`ESTALE`) and directory rename races
vLLM utilizes multiple JIT-compilation engines (FlashInfer, DeepGEMM, OpenAI Triton, and PyTorch TorchInductor) to compile custom GPU kernels during model startup/warmup. 
- By default, these engines write their compiled caches to the user's NFS home directory (`~/.cache/flashinfer/`, `~/.deep_gemm/`, `~/.triton/`, `~/.cache/torchinductor/`).
- **NFS does not support reliable `fcntl.flock()`**, causing FlashInfer compilation lock attempts to fail with `ESTALE` (errno 116).
- On highly concurrent multi-node ranks, concurrent compilation threads write and atomically rename cache directories. On NFS, metadata caching propagation delays lead to ranks attempting to load partially compiled or non-existent cubin files, crashing with:
  ```
  RuntimeError: Assertion error (.../compiler.hpp:147): runtime != nullptr
  ```

### The Solution: Symlink JIT Caches to Lustre (`$SCRATCHDIR`)
Redirecting the caches via environment variables (e.g. `FLASHINFER_JIT_CACHE_DIR`, `DG_JIT_CACHE_DIR`) is **insufficient** because **vLLM's Ray worker agent launcher (`ray_env.py`) strips non-prefixed environment variables**, meaning spawned worker actors revert to default NFS paths.

We resolved this by creating explicit symlinks in the SLURM preamble on all nodes before any Ray actor starts:
```bash
# FlashInfer JIT Cache
export FLASHINFER_JIT_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/flashinfer_cache
mkdir -p "$FLASHINFER_JIT_CACHE_DIR" ~/.cache
ln -sfn "$FLASHINFER_JIT_CACHE_DIR" ~/.cache/flashinfer

# DeepGEMM JIT Cache
export DG_JIT_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/deep_gemm_cache
mkdir -p "$DG_JIT_CACHE_DIR"
ln -sfn "$DG_JIT_CACHE_DIR" ~/.deep_gemm

# Triton JIT Cache
export TRITON_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/triton_cache
mkdir -p "$TRITON_CACHE_DIR"
ln -sfn "$TRITON_CACHE_DIR" ~/.triton

# TorchInductor JIT Cache
export TORCHINDUCTOR_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/torchinductor_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" ~/.cache
ln -sfn "$TORCHINDUCTOR_CACHE_DIR" ~/.cache/torchinductor
```
**Benefits:** Lustre scratch space supports POSIX flock, is lightning fast, eliminates NFS metadata races, and caches persist across runs (preventing a ~25-minute `fused_moe` recompile on every restart).

---

## 2. Custom All-Reduce CUDA Errors Over Slingshot

### The Pitfall: P2P IPC Handshaking Failures
vLLM utilizes its own custom all-reduce kernel for intra-node Tensor Parallel communication to bypass NCCL overhead. However:
- This custom all-reduce relies on CUDA IPC / symmetric memory handles which require peer-to-peer (P2P) GPU memory access.
- Across distinct compute nodes connected via HPE Slingshot, P2P memory access is physically unavailable.
- If vLLM attempts to use custom all-reduce in a multi-node topology, the communication handshake fails and crashes with:
  ```
  Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:434 'invalid argument'
  ```

### The Solution: Disable Custom All-Reduce in Multi-Node Preamble
We force vLLM to use standard, fully supported NCCL collectives across the network by exporting the bypass variable:
```bash
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1
```
This, combined with NCCL tuning parameters for Slingshot:
```bash
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_CROSS_NIC=1
export NCCL_FORCE_FLUSH=0
```
ensures robust, error-free multi-node distributed communication.

---

## 3. Environment Variable & Path Stripping on Workers

### The Pitfall: Binary and venv path mismatch
When starting worker processes via remote `srun` (such as Ray cluster initialization), the remote processes run in non-login environments and do not inherit login-node settings or binary paths. This leads to `ray: command not found` or `nvcc: command not found` errors, blocking kernel compilation.

### The Solution: Standardize Non-Login Wrapper Invocation
All Ray head and worker startup commands must be wrapped in `bash -c` and explicitly source the versioned virtual environment:
```bash
srun --nodelist "$WORKER" --nodes=1 --gpus=4 --mem=0 --cpus-per-task 72 --ntasks-per-node 1 \
  bash -c "source /projects/b6ax/ivllm/0.22.0/bin/activate && \
  VLLM_HOST_IP=$WORKER_IP ray start --block --address=$HEAD_NODE_IP:$RAY_PORT --node-ip-address=$WORKER_IP --object-store-memory=$RAY_OBJECT_STORE_MEMORY" &
```

---

## 4. Multi-User Directory Permissions & HF Offline Mode

### The Pitfall: Blocked Model Cache access and HF 429 Errors
In shared research team directory structures (e.g. `$PROJECTDIR/` / `/projects/` folder):
1. **umask 0022:** Standard umask creates directories and downloaded HuggingFace model cache weights with `drwxr-xr-x` permissions. A second user in the same project group trying to access or write to this cache hits `Permission denied`.
2. **HF API 429s:** Concurrent ranks query the HF hub to verify weight freshness even if the model is fully cached, triggering API rate-limiting blocks.

### The Solution: Enforce Group-Writable Umask and Offline Mode
- **umask 0002:** Prepend `umask 0002` to the login pre-downloads, bare-metal setups, and SLURM inference headers, combined with an explicit `chmod g+w` on the root model cache directories.
- **HF_HUB_OFFLINE=1:** Always download the model fully on a LOGIN node prior to job submission, then launch the inference SLURM script with:
  ```bash
  export HF_HUB_OFFLINE=1
  ```
  This completely prevents API calls at inference startup, saving several minutes and avoiding rate limits.

---

## 5. Memory Pressure Mitigation

### Ray Object Store Overhead
By default, Ray automatically configures its object store size relative to the total node RAM. On Isambard nodes with large host RAM, this consumes a massive footprint, leaving insufficient room for vLLM core engines.
- **Solution:** Cap Ray's object store memory to `RAY_OBJECT_STORE_MEMORY=64GB` (or less) during `ray start --object-store-memory=...`.

---

## Summary of Optimal SLURM Preamble for Isambard AI

Below is the definitive environment configuration block recommended for any multi-node vLLM deployment on Isambard AI Phase 2:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive

# Enforce group-write capabilities for shared weights/caches
umask 0002

# Load necessary host compiler and NCCL dependencies
module load brics/nccl gcc-native

# NVHPC 26.3 CUDA 12.9 Forward Compatibility Preamble
export NVHPC_ROOT=/projects/<your-project>/ivllm/nvhpc/Linux_aarch64/26.3
export CUDA_HOME=$NVHPC_ROOT/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$NVHPC_ROOT/math_libs/12.9/include:$CPATH
export LD_LIBRARY_PATH=$CUDA_HOME/compat:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=gcc
export CXX=g++

# Symlink all major JIT engines to Lustre scratch to bypass NFS locking/metadata races
export FLASHINFER_JIT_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/flashinfer_cache
mkdir -p "$FLASHINFER_JIT_CACHE_DIR" ~/.cache
ln -sfn "$FLASHINFER_JIT_CACHE_DIR" ~/.cache/flashinfer

export DG_JIT_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/deep_gemm_cache
mkdir -p "$DG_JIT_CACHE_DIR"
ln -sfn "$DG_JIT_CACHE_DIR" ~/.deep_gemm

export TRITON_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/triton_cache
mkdir -p "$TRITON_CACHE_DIR"
ln -sfn "$TRITON_CACHE_DIR" ~/.triton

export TORCHINDUCTOR_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/torchinductor_cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" ~/.cache
ln -sfn "$TORCHINDUCTOR_CACHE_DIR" ~/.cache/torchinductor

# Network Tuning for Multi-Node over HPE Slingshot
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_CROSS_NIC=1
export NCCL_FORCE_FLUSH=0
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1 # Crucial to avoid 'invalid argument' CUDA IPC crashes

# Prevent HF API rate-limits during multi-gpu concurrent launch
export HF_HOME=/projects/<your-project>/hf
export HF_HUB_OFFLINE=1
```

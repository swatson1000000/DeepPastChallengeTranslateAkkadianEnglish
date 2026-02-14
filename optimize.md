# NVIDIA GB10 (Blackwell) Optimization Guide

Reference for PyTorch training/inference on the NVIDIA GB10 desktop GPU.
Applicable to any project using this hardware.

## Hardware Specs

| Spec | Value |
|------|-------|
| Architecture | Blackwell (sm_121, compute capability 12.1) |
| SMs | 48 |
| Memory | 128 GB unified LPDDR5X |
| Memory bandwidth | 273 GB/s (12× slower than HBM3 on H100) |
| BF16 tensor cores | 92 TFLOPS |
| FP32 | ~46 TFLOPS |
| Max threads/SM | 1,536 |
| Optimal block size | 256 threads → 100% occupancy (6 blocks/SM × 256 = 1,536) |

## Key Optimization Principles

### 1. Memory bandwidth is the bottleneck
LPDDR5X at 273 GB/s is ~12× slower than H100's HBM3 (~3,350 GB/s).
Every unnecessary memory round-trip hurts disproportionately.
**Fused kernels are critical** — they keep data in registers/shared memory instead of writing back to DRAM between operations.

### 2. Use BF16, never FP16 (for models sensitive to range)
- BF16 has the same exponent range as FP32 (8 exponent bits) → no overflow/NaN issues
- FP16 has only 5 exponent bits → causes NaN in models like ByT5 and some LLMs
- BF16 tensor cores deliver 92 TFLOPS (~2× FP32 throughput)
- Use `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`

### 3. PyTorch native SDPA > flash-attention
- PyTorch's built-in Scaled Dot-Product Attention already targets sm_121
- Faster than the third-party `flash-attn` package on this GPU
- Enable all backends and let PyTorch auto-select:
  ```python
  torch.backends.cuda.enable_flash_sdp(True)
  torch.backends.cuda.enable_mem_efficient_sdp(True)
  torch.backends.cuda.enable_math_sdp(True)
  ```

### 4. torch.compile for kernel fusion
- Fuses elementwise ops, reductions, and attention patterns into single kernels
- Reduces memory traffic (the main bottleneck on GB10)
- Custom CUDA kernels compile fine for compute_120/sm_121
- Apply after moving model to device:
  ```python
  model = model.to(device)
  model = torch.compile(model)
  ```
- First batch is slow (compilation), subsequent batches are faster

### 5. cuDNN benchmark mode
- Enable for workloads with consistent tensor shapes (e.g., fixed sequence length):
  ```python
  torch.backends.cudnn.benchmark = True
  ```

## PyTorch Boilerplate

```python
import torch

# --- GPU setup ---
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda")
model = model.to(device)
model = torch.compile(model)  # fused kernels

# --- Training loop with BF16 autocast ---
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

for batch in dataloader:
    with autocast_ctx:
        outputs = model(**batch)
        loss = outputs.loss
    loss.backward()  # autocast handles grad scaling automatically for BF16
    optimizer.step()
    optimizer.zero_grad()
```

> **Note**: BF16 does NOT need `GradScaler` (unlike FP16). The full exponent range
> means gradients don't underflow, so `loss.backward()` works directly.

## Quick Checklist

- [ ] `--bf16` flag for BF16 autocast (never fp16 for sensitive models)
- [ ] `--compile` flag for `torch.compile` fused kernels
- [ ] SDPA backends all enabled (PyTorch native, not flash-attn package)
- [ ] cuDNN benchmark ON for fixed-shape workloads
- [ ] Gradient checkpointing if memory-limited (trades compute for VRAM)
- [ ] 256-thread CUDA blocks in any custom kernels (100% occupancy)
- [ ] Profile with `torch.profiler` to find remaining memory-bound ops

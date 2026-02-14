"""
GB10 / CUDA GPU optimization utilities.

Provides shared configuration for BF16 autocast, SDPA backend selection,
torch.compile, and hardware diagnostics. Designed for the NVIDIA GB10
(sm_121, 128 GB LPDDR5X, 273 GB/s bandwidth, 92 TFLOPS BF16).

Key insights (CyberBrown):
    - PyTorch native SDPA is faster than flash-attention on sm_121
    - Fused kernels are critical — memory bandwidth is the bottleneck
      (273 GB/s LPDDR5X vs ~3,350 GB/s HBM3 on H100)
    - BF16 tensor cores available (92 TFLOPS) — safe for ByT5 unlike FP16
    - 256-thread CUDA blocks give 100% occupancy (48 SMs × 1536 threads/SM)
"""

import logging
import torch

logger = logging.getLogger(__name__)


def log_gpu_info() -> None:
    """Log detailed GPU hardware info including GB10-specific diagnostics."""
    if not torch.cuda.is_available():
        logger.info("CUDA: not available — running on CPU")
        return

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    cc = f"{props.major}.{props.minor}"

    logger.info(f"GPU: {props.name}")
    logger.info(f"  Compute capability: sm_{props.major}{props.minor} (cc {cc})")
    logger.info(f"  Total memory: {props.total_memory / 1e9:.1f} GB")
    logger.info(f"  SMs: {props.multi_processor_count}")
    logger.info(f"  PyTorch CUDA: {torch.version.cuda}")

    # GB10 detection (compute capability 12.0 / 12.1)
    if props.major >= 12:
        logger.info("  ** GB10 / Blackwell detected **")
        logger.info("  BF16 tensor cores: available (92 TFLOPS)")
        logger.info("  Memory type: LPDDR5X (273 GB/s — bandwidth-limited)")
        logger.info("  Recommendation: use --bf16 --compile for best throughput")


def configure_sdpa() -> None:
    """Configure PyTorch SDPA backends for optimal performance.

    On GB10 (sm_121), PyTorch's native SDPA is faster than flash-attention.
    We enable all backends and let PyTorch choose the fastest at runtime.
    """
    if not torch.cuda.is_available():
        return

    try:
        # Enable all SDPA backends — PyTorch auto-selects the fastest
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("SDPA backends: flash=ON, mem_efficient=ON, math=ON")
    except AttributeError:
        # Older PyTorch without fine-grained SDPA controls
        logger.info("SDPA backends: using PyTorch defaults (fine-grained control unavailable)")


def configure_cudnn() -> None:
    """Enable cuDNN benchmark mode for consistent input sizes."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark: ON")


def check_bf16_support() -> bool:
    """Check if the current GPU supports BF16.

    Returns:
        True if BF16 is supported (Ampere+, compute capability ≥ 8.0).
    """
    if not torch.cuda.is_available():
        return False

    return torch.cuda.is_bf16_supported()


def compile_model(model: torch.nn.Module, enable: bool = True) -> torch.nn.Module:
    """Optionally apply torch.compile for kernel fusion.

    On GB10, fused kernels reduce memory round-trips which is critical
    given the bandwidth-limited LPDDR5X (273 GB/s vs ~3,350 GB/s HBM3).

    Args:
        model: The PyTorch model to compile.
        enable: Whether to actually compile (controlled by --compile flag).

    Returns:
        The (optionally compiled) model.
    """
    if not enable:
        logger.info("torch.compile: OFF")
        return model

    try:
        compiled = torch.compile(model)
        logger.info("torch.compile: ON (fused kernels enabled)")
        return compiled
    except Exception as e:
        logger.warning(f"torch.compile failed ({e}), falling back to eager mode")
        return model


def get_autocast_context(bf16: bool = False):
    """Return the appropriate autocast context manager.

    BF16 is safe for ByT5 (same exponent range as FP32, unlike FP16 which
    overflows and causes NaN). Provides ~2x throughput on GB10's BF16 tensor
    cores (92 TFLOPS BF16 vs ~46 TFLOPS FP32).

    Args:
        bf16: Whether to enable BF16 autocast.

    Returns:
        A context manager for autocast (or nullcontext if bf16=False).
    """
    if bf16 and torch.cuda.is_available() and check_bf16_support():
        logger.info("Autocast: BF16 enabled (safe for ByT5 — same exponent as FP32)")
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif bf16:
        logger.warning("BF16 requested but not supported — falling back to FP32")

    return _nullcontext()


class _nullcontext:
    """Minimal no-op context manager (avoids importing contextlib)."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def setup_gpu(bf16: bool = False, compile_model_flag: bool = False) -> dict:
    """One-call GPU setup: log info, configure SDPA, cuDNN, check BF16.

    Args:
        bf16: Whether BF16 was requested.
        compile_model_flag: Whether torch.compile was requested.

    Returns:
        Dict with 'device', 'bf16_available', 'autocast_ctx' keys.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_gpu_info()
    configure_sdpa()
    configure_cudnn()

    bf16_available = check_bf16_support() if bf16 else False
    if bf16 and not bf16_available:
        logger.warning("BF16 requested but not supported by this GPU")

    return {
        "device": device,
        "bf16_available": bf16_available,
    }

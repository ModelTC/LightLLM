"""JIT launcher and kernel selection for Hopper NVFP4 tile-local decode."""

import dataclasses
import functools
import hashlib
import os
from pathlib import Path

import torch


@functools.lru_cache(None)
def _has_repack_support():
    from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR

    source = (FLASHINFER_INCLUDE_DIR / "flashinfer/attention/prefill.cuh").read_text()
    return "repack_kv_tile_to_16b" in source and "ENABLE_FP4_REPACK_" in source


@functools.lru_cache(None)
def _get_module(dtype, head_dim):
    from flashinfer.jit import gen_batch_prefill_module

    header = Path(__file__).with_suffix(".cuh")
    source_hash = hashlib.sha256(header.read_bytes()).hexdigest()[:12]
    spec = gen_batch_prefill_module(
        "fa2", dtype, torch.uint8, dtype, torch.int32, head_dim, head_dim, 0, False, False, False
    )
    spec = dataclasses.replace(
        spec,
        name=f"{spec.name}_lightllm_repack16_{source_hash}",
        extra_cuda_cflags=[*(spec.extra_cuda_cflags or []), "-include", str(header)],
    )
    return spec.build_and_load()


def get_nvfp4_decode_module(dtype, head_dim):
    """Return the small-tile module, or None when the native module is required."""
    if os.environ.get("FLASHINFER_DISABLE_JIT"):
        return None
    if torch.cuda.get_device_capability() != (9, 0) or not _has_repack_support():
        return None
    return _get_module(dtype, head_dim)

"""Shared RoPE caches and rotary operators."""

import math
import os
from functools import partial
from typing import Callable, Optional

import torch

from lightllm.common.layers.rope.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class RotaryEmbedding:
    """Own the frequency caches and an implementation selected at initialization.

    Gather positions once per batch, then reuse the resulting cos/sin across layers.
    The rotary implementation updates Q/K in place and does not gather positions.
    """

    def __init__(self, cos_cached: torch.Tensor, sin_cached: torch.Tensor, rotary_impl: Callable = rotary_emb_fwd):
        self.cos_cached = cos_cached
        self.sin_cached = sin_cached
        self._forward = rotary_impl

    def get_cos_sin(self, position_ids: torch.Tensor):
        shape = (*position_ids.shape, self.cos_cached.shape[-1])
        positions = position_ids.reshape(-1)
        cos = torch.index_select(self.cos_cached, 0, positions).view(shape)
        sin = torch.index_select(self.sin_cached, 0, positions).view(shape)
        return cos, sin

    def __call__(self, q: torch.Tensor, k: Optional[torch.Tensor], cos: torch.Tensor, sin: torch.Tensor):
        return self._forward(q=q, k=k, cos=cos, sin=sin)


def get_rope_type(config: dict) -> str:
    rope_scaling = config.get("rope_scaling")
    if rope_scaling is None:
        return "default"
    if "rope_type" in rope_scaling:
        return rope_scaling["rope_type"]
    if "type" in rope_scaling:
        return rope_scaling["type"]
    raise ValueError(f"Unknown RoPE scaling format {rope_scaling}")


def get_rope(
    config: dict,
    head_dim: int,
    max_seq_length: int,
    data_type: torch.dtype,
    device: torch.device,
    *,
    rotary_impl: Optional[Callable] = None,
    mrope_section=None,
    mrope_interleaved: bool = False,
) -> RotaryEmbedding:
    rope_type = get_rope_type(config)
    rope_init_functions = {
        "default": get_default_rope,
        "mrope": get_default_rope,
        "yarn": get_yarn_rope,
        "dynamic": get_dynamic_ntk_rope,
        "su": get_su_rope,
        "llama3": get_llama3_rope,
    }
    if rope_type not in rope_init_functions:
        raise ValueError(f"Unknown RoPE scaling type {rope_type}")
    cos, sin = rope_init_functions[rope_type](config, head_dim, max_seq_length, data_type, device)
    partial_rotary_factor = config.get("partial_rotary_factor", 1.0)
    # MRoPE's position layout is independent of the frequency scaling algorithm.
    if mrope_section is not None:
        from lightllm.common.layers.rope.triton_kernel.mrope import mrope_triton_fused

        rotary_impl = partial(
            rotary_impl or mrope_triton_fused,
            mrope_section=torch.tensor(mrope_section, dtype=torch.int32, device=device),
            is_interleaved=mrope_interleaved,
        )
    else:
        rotary_impl = rotary_impl or rotary_emb_fwd
    return RotaryEmbedding(cos, sin, partial(rotary_impl, partial_rotary_factor=partial_rotary_factor))


def get_default_rope(config, head_dim, max_seq_length, data_type, device):
    partial_head_dim = int(config.get("partial_rotary_factor", 1) * head_dim)
    rope_scaling = config.get("rope_scaling") or {}
    rope_scaling_factor = rope_scaling.get("factor", 1.0)

    base = config.get("rope_theta", 10000.0)

    if "max_sequence_length" in config:
        max_seq_len = config["max_sequence_length"]
    else:
        max_position_embeddings = config.get("max_position_embeddings", 2048 if base <= 10000.0 + 1e-5 else 16384)
        max_seq_len = max_position_embeddings * rope_scaling_factor

    # NTK
    try:
        ntk_alpha = float(os.environ.get("LIGHTLLM_NTK_ALPHA", 1))
        assert ntk_alpha >= 1
        if ntk_alpha > 1:
            logger.info(f"Note: NTK enabled, alpha set to {ntk_alpha}")
        max_seq_len *= ntk_alpha
        base = base * (ntk_alpha ** (partial_head_dim / (partial_head_dim - 2)))  # Base change formula
    except (ValueError, AssertionError, ZeroDivisionError):
        pass

    inv_freq = 1.0 / (
        base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
    )
    t = (
        torch.arange(max(max_seq_len + 1024 * 128, max_seq_length), device="cpu", dtype=torch.float32)
        / rope_scaling_factor
    )
    freqs = torch.outer(t, inv_freq)

    cos_cached = torch.cos(freqs).to(data_type).to(device)
    sin_cached = torch.sin(freqs).to(data_type).to(device)
    return cos_cached, sin_cached


def get_dynamic_ntk_rope(config, head_dim, max_seq_length, data_type, device):
    partial_head_dim = int(config.get("partial_rotary_factor", 1) * head_dim)
    max_position_embeddings = config.get("max_position_embeddings", 2048)
    base = config.get("rope_theta", 10000.0)
    rope_scaling = config.get("rope_scaling") or {}
    scaling_factor = rope_scaling.get("factor", 1.0)
    max_seq_len = max(max_seq_length, max_position_embeddings)
    cos_cached = torch.zeros((max_seq_len, partial_head_dim // 2), dtype=data_type, device=device)
    sin_cached = torch.zeros((max_seq_len, partial_head_dim // 2), dtype=data_type, device=device)

    inv_freq = 1.0 / (
        base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
    )
    t = torch.arange(max_position_embeddings, device="cpu", dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_cached[0:max_position_embeddings, :] = torch.cos(freqs).to(data_type).to(device)
    sin_cached[0:max_position_embeddings, :] = torch.sin(freqs).to(data_type).to(device)

    for seq_loc_index in range(max_position_embeddings, max_seq_len):
        new_base = base * ((scaling_factor * (seq_loc_index + 1) / max_position_embeddings) - (scaling_factor - 1)) ** (
            partial_head_dim / (partial_head_dim - 2)
        )
        inv_freq = 1.0 / (
            new_base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
        )
        t = torch.tensor([seq_loc_index], device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos_cached[seq_loc_index : seq_loc_index + 1, :] = torch.cos(freqs).to(data_type).to(device)
        sin_cached[seq_loc_index : seq_loc_index + 1, :] = torch.sin(freqs).to(data_type).to(device)
    return cos_cached, sin_cached


def get_yarn_rope(config, head_dim, max_seq_length, data_type, device):
    from .yarn_rotary_utils import find_correction_range, linear_ramp_mask, get_mscale

    dim = int(config.get("partial_rotary_factor", 1.0) * head_dim)
    max_position_embeddings = config.get("max_position_embeddings", 2048)
    base = config.get("rope_theta", 10000.0)
    rope_scaling = config.get("rope_scaling") or {}
    scale = rope_scaling.get("factor", 1.0)
    original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 2048)
    beta_fast = 32.0
    beta_slow = 1.0

    pos_freqs = base ** (torch.arange(0, dim, 2).float().to(device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scale * pos_freqs)
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
    inv_freq_mask = 1 - linear_ramp_mask(low, high, dim // 2).float().to(device)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

    mscale = get_mscale(scale)

    t = torch.arange(max(max_position_embeddings, max_seq_length), device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # Rotary kernels reuse each frequency for a pair of channels, so cache only half the rotary dimension.
    cos_cached = (freqs.cos() * mscale).to(data_type)
    sin_cached = (freqs.sin() * mscale).to(data_type)

    return cos_cached, sin_cached


def get_su_rope(config, head_dim, max_seq_length, data_type, device):
    rope_scaling = config["rope_scaling"]
    short_factor = rope_scaling["short_factor"]
    long_factor = rope_scaling["long_factor"]
    original_max_position_embeddings = config["original_max_position_embeddings"]
    max_position_embeddings = config.get("max_position_embeddings", original_max_position_embeddings)
    base = config.get("rope_theta", 10000.0)
    short_factor = torch.tensor(short_factor, dtype=torch.float32, device="cpu")
    long_factor = torch.tensor(long_factor, dtype=torch.float32, device="cpu")

    scale = max_position_embeddings / original_max_position_embeddings
    if scale <= 1.0:
        rope_scaling_factor = 1.0
    else:
        rope_scaling_factor = math.sqrt(1 + math.log(scale) / math.log(original_max_position_embeddings))

    max_seq_len = max(max_seq_length, max_position_embeddings)
    cos_cached = torch.zeros((max_seq_len, head_dim // 2), dtype=data_type, device=device)
    sin_cached = torch.zeros((max_seq_len, head_dim // 2), dtype=data_type, device=device)

    inv_freq = 1.0 / (
        short_factor * base ** (torch.arange(0, head_dim, 2, device="cpu", dtype=torch.float32) / head_dim)
    )
    t = torch.arange(original_max_position_embeddings, device="cpu", dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_cached[0:original_max_position_embeddings, :] = (
        (torch.cos(freqs) * rope_scaling_factor).to(data_type).to(device)
    )
    sin_cached[0:original_max_position_embeddings, :] = (
        (torch.sin(freqs) * rope_scaling_factor).to(data_type).to(device)
    )

    inv_freq = 1.0 / (
        long_factor * base ** (torch.arange(0, head_dim, 2, device="cpu", dtype=torch.float32) / head_dim)
    )
    t = torch.arange(original_max_position_embeddings, max_seq_len, device="cpu", dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_cached[original_max_position_embeddings:, :] = (torch.cos(freqs) * rope_scaling_factor).to(data_type).to(device)
    sin_cached[original_max_position_embeddings:, :] = (torch.sin(freqs) * rope_scaling_factor).to(data_type).to(device)

    return cos_cached, sin_cached


def get_llama3_rope(config, head_dim, max_seq_length, data_type, device):
    partial_head_dim = int(config.get("partial_rotary_factor", 1) * head_dim)
    base = config.get("rope_theta", 10000.0)

    rope_scaling = config.get("rope_scaling") or {}
    scale_factor = rope_scaling.get("factor", 8.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    origin_context_len = rope_scaling.get("original_max_position_embeddings", 8192)

    max_seq_len = config.get("max_position_embeddings", 2048)

    inv_freq = 1.0 / (
        base ** (torch.arange(0, partial_head_dim, 2, device="cpu", dtype=torch.float32) / partial_head_dim)
    )

    low_freq_wavelen = origin_context_len / low_freq_factor
    high_freq_wavelen = origin_context_len / high_freq_factor
    new_inv_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_inv_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_inv_freqs.append(freq / scale_factor)
        else:
            smooth = (origin_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_inv_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    inv_freq = torch.tensor(new_inv_freqs, dtype=torch.float32, device="cpu")

    t = torch.arange(max(max_seq_len, max_seq_length), device="cpu", dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)

    cos_cached = torch.cos(freqs).to(data_type).to(device)
    sin_cached = torch.sin(freqs).to(data_type).to(device)
    return cos_cached, sin_cached

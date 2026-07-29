import threading
from typing import Dict, Tuple

import torch


_slot_lock = threading.Lock()
_slot_pools: Dict[Tuple[int, torch.dtype, int, int, int], torch.Tensor] = {}


def create_moonep_source_weight(
    local_num_experts: int,
    out_dim: int,
    in_dim: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    try:
        from moonep.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment
    except ImportError as exc:
        raise RuntimeError(
            "MoonEP is required for --moe_ep_backend moonep. " "Install it from https://github.com/MoonshotAI/MoonEP."
        ) from exc

    chunk_shape = [local_num_experts, out_dim, in_dim]
    padded_num_experts = pad_dim0_for_alignment(chunk_shape, dtype)
    if padded_num_experts != local_num_experts:
        raise ValueError(
            "MoonEP expert weights do not align to the CUDA VMM granularity: "
            f"local experts={local_num_experts}, padded experts={padded_num_experts}, "
            f"shape=({out_dim}, {in_dim})."
        )
    return create_nvl_dist_tensor(
        chunk_shape=chunk_shape,
        dtype=dtype,
        local_rank=rank,
        world_size=world_size,
    )


def get_moonep_prefetch_slots(
    num_slots: int,
    out_dim: int,
    in_dim: int,
    dtype: torch.dtype,
    device_id: int,
) -> torch.Tensor:
    key = (device_id, dtype, num_slots, out_dim, in_dim)
    with _slot_lock:
        slots = _slot_pools.get(key)
        if slots is None:
            slots = torch.empty(
                (num_slots, out_dim, in_dim),
                dtype=dtype,
                device=f"cuda:{device_id}",
            )
            _slot_pools[key] = slots
        return slots

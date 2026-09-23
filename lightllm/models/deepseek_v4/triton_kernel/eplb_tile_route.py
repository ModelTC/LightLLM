"""Default-off DeepSeek-V4 full-layout 128-token EPLB tile routing."""
import functools
import hashlib
import os

import torch
import torch.distributed as dist

from lightllm.utils.envs_utils import get_dsv4_eplb_tile_routing


@functools.lru_cache(maxsize=1)
def _load_cuda():
    from torch.utils.cpp_extension import load

    source_path = os.path.join(os.path.dirname(__file__), "csrc", "eplb_tile_route.cu")
    flags = ["-O3"]
    with open(source_path, "rb") as source_file:
        source = source_file.read()
    capability = torch.cuda.get_device_capability()
    cache_key = b"\0".join(
        [
            source,
            " ".join(flags).encode(),
            torch.__version__.encode(),
            str(torch.version.cuda).encode(),
            f"sm{capability[0]}{capability[1]}".encode(),
            os.environ.get("TORCH_CUDA_ARCH_LIST", "").encode(),
        ]
    )
    return load(
        name=f"lightllm_dsv4_eplb_tile_route_v1_{hashlib.sha256(cache_key).hexdigest()[:16]}",
        sources=[source_path],
        extra_cuda_cflags=flags,
        verbose=False,
    )


def tile_routing_peak_nbytes(max_tokens: int, topk: int) -> int:
    """Peak allocation excluding the caller-owned physical-id output."""
    if max_tokens < 0 or topk <= 0:
        raise ValueError("max_tokens/topk must be positive")
    routes = max_tokens * topk
    return routes * (8 + 4) + 256 * 4 + 8 * 256 * 4 + 256 * 8 * 4 + 3 * 4


def _check(logical_ids, physical_output, mapping, replicas, physical_slots_per_rank, source_rank):
    if not get_dsv4_eplb_tile_routing():
        raise RuntimeError("tile routing is disabled")
    tensors = (
        (logical_ids, torch.int64, "logical_ids"),
        (physical_output, torch.int64, "physical_output"),
        (mapping, torch.int32, "logical_to_physical"),
        (replicas, torch.int32, "replica_counts"),
    )
    for tensor, dtype, name in tensors:
        if not tensor.is_cuda or not tensor.is_contiguous() or tensor.dtype != dtype:
            raise ValueError(f"{name} must be contiguous CUDA {dtype}")
        if tensor.device != logical_ids.device:
            raise ValueError("tile-routing tensors must share a CUDA device")
    if logical_ids.shape != physical_output.shape or mapping.ndim != 2 or mapping.shape[0] != 256:
        raise ValueError("invalid logical/output/map shape")
    if tuple(replicas.shape) != (256,) or not 0 <= source_rank < 8:
        raise ValueError("invalid replica counts or source rank")
    if physical_slots_per_rank not in (32, 33, 34) or mapping.shape[1] != 8:
        raise ValueError("tile routing requires an [256, 8] map and physical slots 32..34")
    if not dist.is_initialized() or dist.get_backend() != "nccl" or dist.get_world_size() != 8:
        raise RuntimeError("tile routing requires the default NCCL world group of size 8")
    if source_rank != dist.get_rank():
        raise ValueError("source_rank must match the distributed rank")


def route(
    logical_ids,
    physical_output,
    mapping,
    replica_counts,
    physical_slots_per_rank,
    source_rank,
    alloc_tensor_func=torch.empty,
):
    """Route existing logical top-k ids into the caller-owned physical output.

    All ranks, including M=0 ranks, run this function in the same order.
    No host synchronization is performed; invalid solver or remap state raises
    a device assertion on the current stream rather than falling back.
    """
    _check(logical_ids, physical_output, mapping, replica_counts, physical_slots_per_rank, source_rank)
    module = _load_cuda()
    flat = logical_ids.reshape(-1)
    histogram, ordinal = module.count(flat)
    gathered = alloc_tensor_func((8, 256), dtype=torch.int32, device=flat.device)
    dist.all_gather_into_tensor(gathered, histogram)
    quota, status, maximum, augmentations = module.solve(
        gathered, mapping, replica_counts, mapping.shape[1], int(physical_slots_per_rank)
    )
    routed = module.remap(
        flat,
        ordinal,
        gathered,
        quota,
        status,
        physical_output.reshape(-1),
        mapping,
        replica_counts,
        mapping.shape[1],
        int(physical_slots_per_rank),
        int(source_rank),
    )
    # Status is consumed by remap on device; invalid state asserts on its stream.
    _ = routed, status, maximum, augmentations
    return physical_output

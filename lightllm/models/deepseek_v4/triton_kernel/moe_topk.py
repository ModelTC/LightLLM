import functools
import hashlib
import os

import torch


@torch.no_grad()
def deepseek_v4_eplb_topk(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    input_tokens: torch.Tensor | None,
    hash_indices_table: torch.Tensor | None,
    topk: int,
    routed_scaling_factor: float,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_counter: torch.Tensor,
    sample_index: int,
    record_load: bool,
    alloc_tensor_func=torch.empty,
    return_logical_ids: bool = False,
    weights_out: torch.Tensor | None = None,
    physical_ids_out: torch.Tensor | None = None,
    logical_ids_out: torch.Tensor | None = None,
):
    """Select DeepSeek-V4 routes and emit EPLB physical expert IDs."""
    token_num, num_experts = logits.shape
    if num_experts != 256:
        raise RuntimeError(f"DeepSeek-V4 EPLB fused top-k requires 256 experts, got {num_experts}")
    if hash_indices_table is not None:
        topk = hash_indices_table.shape[1]
    weights = weights_out
    if weights is None:
        weights = alloc_tensor_func((token_num, topk), dtype=torch.float32, device=logits.device)
    physical_ids = physical_ids_out
    if physical_ids is None:
        physical_ids = alloc_tensor_func((token_num, topk), dtype=torch.long, device=logits.device)
    logical_ids = logical_ids_out if return_logical_ids else None
    if return_logical_ids and logical_ids is None:
        logical_ids = alloc_tensor_func((token_num, topk), dtype=torch.long, device=logits.device)
    if token_num == 0:
        return weights, physical_ids, logical_ids
    _load_cuda().moe_topk_eplb(
        logits.contiguous(),
        bias.contiguous() if bias is not None else logits,
        input_tokens.contiguous() if input_tokens is not None else physical_ids,
        hash_indices_table.contiguous() if hash_indices_table is not None else physical_ids,
        weights,
        physical_ids,
        logical_ids if logical_ids is not None else physical_ids,
        logical_to_physical_map,
        logical_replica_count,
        expert_counter,
        sample_index,
        record_load,
        return_logical_ids,
        hash_indices_table is not None,
        routed_scaling_factor,
    )
    return weights, physical_ids, logical_ids


@functools.lru_cache(maxsize=1)
def _load_cuda():
    from torch.utils.cpp_extension import load

    source_path = os.path.join(os.path.dirname(__file__), "csrc", "moe_topk_eplb.cu")
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
        name=f"lightllm_dsv4_eplb_topk_v1_{hashlib.sha256(cache_key).hexdigest()[:16]}",
        sources=[source_path],
        extra_cuda_cflags=flags,
        verbose=False,
    )

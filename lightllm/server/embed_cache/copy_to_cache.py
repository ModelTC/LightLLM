import torch

import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _offload_embed_tensor_to_cache(
    embed_tensor_ptr,
    gpu_stride0,
    gpu_stride1,
    gpu_stride2,
    cache_tensor_ptr,
    cpu_stride0,
    cpu_stride1,
    cpu_stride2,
    start_index_in_cache,
    layer_num,
    hidden_size,
    BLOCK: tl.constexpr,
):
    token_index = tl.program_id(0).to(tl.int64)
    dest_index = (start_index_in_cache + token_index).to(tl.int64)

    for layer_index in range(layer_num):
        for block_index in range(tl.cdiv(hidden_size, BLOCK)):
            off = block_index * BLOCK + tl.arange(0, BLOCK)
            mask = off < hidden_size
            gpu_data = tl.load(
                embed_tensor_ptr + token_index * gpu_stride0 + layer_index * gpu_stride1 + off * gpu_stride2, mask=mask
            )
            tl.store(
                cache_tensor_ptr + dest_index * cpu_stride0 + layer_index * cpu_stride1 + off * cpu_stride2,
                gpu_data,
                mask=mask,
            )

    return


@torch.no_grad()
def offload_embed_tensor_to_cache(
    embed_tensor: torch.Tensor,
    cache_tensor: torch.Tensor,
    start_index_in_cache: int,
):
    if len(embed_tensor.shape) == 2:
        embed_tensor = embed_tensor.reshape(embed_tensor.shape[0], 1, embed_tensor.shape[1])

    token_num = embed_tensor.shape[0]
    grid = (token_num,)

    _offload_embed_tensor_to_cache[grid](
        embed_tensor_ptr=embed_tensor,
        gpu_stride0=embed_tensor.stride(0),
        gpu_stride1=embed_tensor.stride(1),
        gpu_stride2=embed_tensor.stride(2),
        cache_tensor_ptr=cache_tensor,
        cpu_stride0=cache_tensor.stride(0),
        cpu_stride1=cache_tensor.stride(1),
        cpu_stride2=cache_tensor.stride(2),
        start_index_in_cache=start_index_in_cache,
        layer_num=embed_tensor.shape[1],
        hidden_size=embed_tensor.shape[2],
        BLOCK=256,
        num_warps=4,
        num_stages=1,
    )
    return

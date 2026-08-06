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
    source_layer_num: tl.constexpr,
    cache_layer_num: tl.constexpr,
    hidden_size,
    BLOCK: tl.constexpr,
):
    token_index = tl.program_id(0).to(tl.int64)
    dest_index = (start_index_in_cache + token_index).to(tl.int64)

    # Qwen3-VL/Omni 的 embed cache 为 [token, layer, H]，layer=4：
    # cache[:, 0, :] 为主 embedding，cache[:, 1:4, :] 为 vision deepstack。
    # 音频只有主 embedding（source_layer_num=1），embedding 阶段只读
    # cache[:, 0, :]，本身不受影响；但 transformer 里 apply_deepstack 会对
    # images+audios 的 cache 位置统一叠加 cache[:, 1:4, :]。若 slot 曾被图像
    # 占用，只覆盖第 0 层会留下旧 deepstack，音频 token 就会被错误叠加。
    # 因此按 cache_layer_num 写满：有效源层正常拷贝，超出 source 的层用 0 填充。
    for layer_index in range(cache_layer_num):
        layer_mask = layer_index < source_layer_num
        for block_index in range(tl.cdiv(hidden_size, BLOCK)):
            off = block_index * BLOCK + tl.arange(0, BLOCK)
            mask = off < hidden_size
            gpu_data = tl.load(
                embed_tensor_ptr + token_index * gpu_stride0 + layer_index * gpu_stride1 + off * gpu_stride2,
                mask=mask & layer_mask,
                other=0.0,
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
        # 音频 source=1、vision+deepstack cache=4；kernel 将多出的层写 0。
        source_layer_num=embed_tensor.shape[1],
        cache_layer_num=cache_tensor.shape[1],
        hidden_size=embed_tensor.shape[2],
        BLOCK=256,
        num_warps=4,
        num_stages=1,
    )
    return

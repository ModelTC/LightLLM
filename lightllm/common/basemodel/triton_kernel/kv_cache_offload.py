import torch

import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _offload_gpu_kv_to_cpu(
    token_indexes_ptr,
    gpu_kv_cache_ptr,
    gpu_stride0,
    gpu_stride1,
    gpu_stride2,
    gpu_stride3,
    gpu_kv_cache_scale_ptr,
    gpu_scale_stride0,
    gpu_scale_stride1,
    gpu_scale_stride2,
    gpu_scale_stride3,
    cpu_kv_cache_ptr,
    cpu_stride0,
    cpu_stride1,
    cpu_stride2,
    cpu_stride3,
    cpu_stride4,
    cpu_kv_cache_scale_ptr,
    cpu_scale_stride0,
    cpu_scale_stride1,
    cpu_scale_stride2,
    cpu_scale_stride3,
    cpu_scale_stride4,
    page_indexes_ptr,
    page_readies_ptr,
    layer_num,
    head_dim,
    scale_head_dim,
    block_num,
    cpu_k_start_head_index: tl.constexpr,
    cpu_k_head_num: tl.constexpr,
    gpu_k_start_head_index: tl.constexpr,
    gpu_k_head_num: tl.constexpr,
    cpu_v_start_head_index: tl.constexpr,
    cpu_v_head_num: tl.constexpr,
    gpu_v_start_head_index: tl.constexpr,
    gpu_v_head_num: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    block_start_index = tl.program_id(0)
    block_split_size = tl.num_programs(axis=0)

    for block_index in tl.range(block_start_index, block_num, block_split_size):
        cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)

        ready_state = tl.load(page_readies_ptr + block_index)

        mask_layer_num = tl.where(cpu_page_index == -1, 0, 1)
        mask_layer_num = tl.where(ready_state, 0, mask_layer_num)

        token_range = block_index * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        token_indexes = tl.load(token_indexes_ptr + token_range).to(tl.int64)
        head_dim_range = tl.arange(0, BLOCK_HEAD_DIM)
        head_dim_mask = head_dim_range < head_dim
        scale_head_dim_mask = head_dim_range < scale_head_dim

        for layer_index in range(layer_num * mask_layer_num):
            for k_head_index in range(gpu_k_head_num):
                gpu_k_head_index = k_head_index + gpu_k_start_head_index
                cpu_k_head_index = k_head_index + cpu_k_start_head_index

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + token_indexes[:, None] * gpu_stride1
                    + gpu_k_head_index.to(tl.int64) * gpu_stride2
                    + head_dim_range[None, :]
                )
                gpu_data = tl.load(gpu_ptr, mask=head_dim_mask[None, :], other=0.0)
                cpu_ptr = (
                    cpu_kv_cache_ptr
                    + cpu_page_index * cpu_stride0
                    + layer_index.to(tl.int64) * cpu_stride1
                    + tl.arange(0, TOKEN_BLOCK)[:, None] * cpu_stride2
                    + cpu_k_head_index * cpu_stride3
                    + head_dim_range[None, :]
                )
                tl.store(
                    cpu_ptr,
                    gpu_data,
                    mask=head_dim_mask[None, :],
                    cache_modifier=".wt",
                )
                if HAS_SCALE:
                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + token_indexes[:, None] * gpu_scale_stride1
                        + gpu_k_head_index.to(tl.int64) * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )
                    gpu_scale_data = tl.load(gpu_scale_ptr, mask=scale_head_dim_mask[None, :], other=0.0)
                    cpu_scale_ptr = (
                        cpu_kv_cache_scale_ptr
                        + cpu_page_index * cpu_scale_stride0
                        + layer_index.to(tl.int64) * cpu_scale_stride1
                        + tl.arange(0, TOKEN_BLOCK)[:, None] * cpu_scale_stride2
                        + cpu_k_head_index * cpu_scale_stride3
                        + head_dim_range[None, :]
                    )
                    tl.store(
                        cpu_scale_ptr,
                        gpu_scale_data,
                        mask=scale_head_dim_mask[None, :],
                        cache_modifier=".wt",
                    )

            for v_head_index in range(gpu_v_head_num):
                gpu_v_head_index = v_head_index + gpu_v_start_head_index
                cpu_v_head_index = v_head_index + cpu_v_start_head_index

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + token_indexes[:, None] * gpu_stride1
                    + gpu_v_head_index.to(tl.int64) * gpu_stride2
                    + head_dim_range[None, :]
                )
                gpu_data = tl.load(gpu_ptr, mask=head_dim_mask[None, :], other=0.0)
                cpu_ptr = (
                    cpu_kv_cache_ptr
                    + cpu_page_index * cpu_stride0
                    + layer_index.to(tl.int64) * cpu_stride1
                    + tl.arange(0, TOKEN_BLOCK)[:, None] * cpu_stride2
                    + cpu_v_head_index * cpu_stride3
                    + head_dim_range[None, :]
                )
                tl.store(
                    cpu_ptr,
                    gpu_data,
                    mask=head_dim_mask[None, :],
                    cache_modifier=".wt",
                )
                if HAS_SCALE:
                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + token_indexes[:, None] * gpu_scale_stride1
                        + gpu_v_head_index.to(tl.int64) * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )
                    gpu_scale_data = tl.load(gpu_scale_ptr, mask=scale_head_dim_mask[None, :], other=0.0)
                    cpu_scale_ptr = (
                        cpu_kv_cache_scale_ptr
                        + cpu_page_index * cpu_scale_stride0
                        + layer_index.to(tl.int64) * cpu_scale_stride1
                        + tl.arange(0, TOKEN_BLOCK)[:, None] * cpu_scale_stride2
                        + cpu_v_head_index * cpu_scale_stride3
                        + head_dim_range[None, :]
                    )
                    tl.store(
                        cpu_scale_ptr,
                        gpu_scale_data,
                        mask=scale_head_dim_mask[None, :],
                        cache_modifier=".wt",
                    )

    return


@torch.no_grad()
def offload_gpu_kv_to_cpu(
    token_indexes: torch.Tensor,
    gpu_kv_cache: torch.Tensor,
    gpu_kv_cache_scale: Optional[torch.Tensor],
    cpu_kv_cache: torch.Tensor,
    cpu_kv_cache_scale: Optional[torch.Tensor],
    page_indexes: torch.Tensor,
    page_readies: torch.Tensor,
    tp_index: int,
    tp_world_size: int,
    grid_num: int,
    _cache_data={},
):
    """
    this function is used to offload GPU KV cache to CPU KV cache.
    Args:
        token_indexes: (token_num,)
        gpu_kv_cache: (layer_num, token_num, head_num, head_dim)
        cpu_kv_cache: (all_page_num, layer_num, token_block_size, head_num, head_dim)
        page_indexes: (page_num,)
        page_readies: (page_num,)
    """
    token_block_size = cpu_kv_cache.shape[2]
    token_num = token_indexes.shape[0]
    assert token_num == page_indexes.shape[0] * token_block_size
    assert page_indexes.shape == page_readies.shape

    gpu_heads = gpu_kv_cache.shape[2]
    gpu_head_dim = gpu_kv_cache.shape[3]
    cpu_heads = cpu_kv_cache.shape[3]
    cpu_head_dim = cpu_kv_cache.shape[4]
    assert gpu_head_dim == cpu_head_dim
    assert gpu_kv_cache.shape[0] == cpu_kv_cache.shape[1]
    head_dim = gpu_head_dim
    scale_size = (tp_world_size * gpu_heads) // cpu_heads

    # 计算需要拷贝的 head 索引的对应关系
    if (gpu_heads, cpu_heads, tp_index, tp_world_size) in _cache_data:
        need_offload, head_info_tuple = _cache_data[(gpu_heads, cpu_heads, tp_index, tp_world_size)]
    else:
        if cpu_heads > 1:
            assert (tp_world_size * gpu_heads) % cpu_heads == 0
            assert cpu_heads % 2 == 0

            cpu_heads_index = (
                torch.arange(0, cpu_heads, device="cpu", dtype=torch.int32)
                .view(cpu_heads, 1)
                .tile((1, scale_size))
                .view(2, tp_world_size, -1)
            )
            # k
            k_cpu_heads_index = cpu_heads_index[0][tp_index]
            # v
            v_cpu_heads_index = cpu_heads_index[1][tp_index]

            cpu_heads_index = torch.cat([k_cpu_heads_index, v_cpu_heads_index], dim=0).view(2, -1).numpy()
            gpu_heads_index = torch.arange(0, gpu_heads, device="cpu", dtype=torch.int32).view(2, -1).numpy()

            need_offload = tp_index % scale_size == 0

            cpu_k_start_head_index = int(cpu_heads_index[0, 0])
            cpu_k_head_num = len(cpu_heads_index[0])
            gpu_k_start_head_index = int(gpu_heads_index[0, 0])
            gpu_k_head_num = len(gpu_heads_index[0])
            assert cpu_k_head_num == gpu_k_head_num
            cpu_v_start_head_index = int(cpu_heads_index[1, 0])
            cpu_v_head_num = len(cpu_heads_index[1])
            gpu_v_start_head_index = int(gpu_heads_index[1, 0])
            gpu_v_head_num = len(gpu_heads_index[1])
            assert cpu_v_head_num == gpu_v_head_num

            head_info_tuple = (
                cpu_k_start_head_index,
                cpu_k_head_num,
                gpu_k_start_head_index,
                gpu_k_head_num,
                cpu_v_start_head_index,
                cpu_v_head_num,
                gpu_v_start_head_index,
                gpu_v_head_num,
            )

        else:
            assert gpu_heads == 1
            assert cpu_heads == 1

            need_offload = tp_index == 0

            cpu_k_start_head_index = 0
            cpu_k_head_num = 1
            gpu_k_start_head_index = 0
            gpu_k_head_num = 1
            cpu_v_start_head_index = 0
            cpu_v_head_num = 0
            gpu_v_start_head_index = 0
            gpu_v_head_num = 0
            head_info_tuple = (
                cpu_k_start_head_index,
                cpu_k_head_num,
                gpu_k_start_head_index,
                gpu_k_head_num,
                cpu_v_start_head_index,
                cpu_v_head_num,
                gpu_v_start_head_index,
                gpu_v_head_num,
            )

        _cache_data[(gpu_heads, cpu_heads, tp_index, tp_world_size)] = (need_offload, head_info_tuple)

    (
        cpu_k_start_head_index,
        cpu_k_head_num,
        gpu_k_start_head_index,
        gpu_k_head_num,
        cpu_v_start_head_index,
        cpu_v_head_num,
        gpu_v_start_head_index,
        gpu_v_head_num,
    ) = head_info_tuple

    if not need_offload:
        return

    assert token_block_size == triton.next_power_of_2(token_block_size)
    page_num = page_indexes.shape[0]

    grid = (grid_num,)
    num_warps = 4
    HAS_SCALE = gpu_kv_cache_scale is not None and cpu_kv_cache_scale is not None
    if HAS_SCALE:
        scale_head_dim = gpu_kv_cache_scale.shape[-1]
        gpu_scale_stride = gpu_kv_cache_scale.stride()
        cpu_scale_stride = cpu_kv_cache_scale.stride()
    else:
        scale_head_dim = 0
        gpu_scale_stride = [0 for _ in range(10)]
        cpu_scale_stride = [0 for _ in range(10)]

    _offload_gpu_kv_to_cpu[grid](
        token_indexes_ptr=token_indexes,
        gpu_kv_cache_ptr=gpu_kv_cache,
        gpu_stride0=gpu_kv_cache.stride(0),
        gpu_stride1=gpu_kv_cache.stride(1),
        gpu_stride2=gpu_kv_cache.stride(2),
        gpu_stride3=gpu_kv_cache.stride(3),
        gpu_kv_cache_scale_ptr=gpu_kv_cache_scale,
        gpu_scale_stride0=gpu_scale_stride[0],
        gpu_scale_stride1=gpu_scale_stride[1],
        gpu_scale_stride2=gpu_scale_stride[2],
        gpu_scale_stride3=gpu_scale_stride[3],
        cpu_kv_cache_ptr=cpu_kv_cache,
        cpu_stride0=cpu_kv_cache.stride(0),
        cpu_stride1=cpu_kv_cache.stride(1),
        cpu_stride2=cpu_kv_cache.stride(2),
        cpu_stride3=cpu_kv_cache.stride(3),
        cpu_stride4=cpu_kv_cache.stride(4),
        cpu_kv_cache_scale_ptr=cpu_kv_cache_scale,
        cpu_scale_stride0=cpu_scale_stride[0],
        cpu_scale_stride1=cpu_scale_stride[1],
        cpu_scale_stride2=cpu_scale_stride[2],
        cpu_scale_stride3=cpu_scale_stride[3],
        cpu_scale_stride4=cpu_scale_stride[4],
        page_indexes_ptr=page_indexes,
        page_readies_ptr=page_readies,
        layer_num=gpu_kv_cache.shape[0],
        head_dim=head_dim,
        scale_head_dim=scale_head_dim,
        block_num=page_num,
        cpu_k_start_head_index=cpu_k_start_head_index,
        cpu_k_head_num=cpu_k_head_num,
        gpu_k_start_head_index=gpu_k_start_head_index,
        gpu_k_head_num=gpu_k_head_num,
        cpu_v_start_head_index=cpu_v_start_head_index,
        cpu_v_head_num=cpu_v_head_num,
        gpu_v_start_head_index=gpu_v_start_head_index,
        gpu_v_head_num=gpu_v_head_num,
        BLOCK_HEAD_DIM=triton.next_power_of_2(head_dim),
        TOKEN_BLOCK=token_block_size,
        HAS_SCALE=HAS_SCALE,
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _load_cpu_cache_to_gpu(
    gpu_mem_indexes_ptr,
    copy_token_num,
    copy_block_num,
    cpu_mem_indexes_ptr,
    cpu_page_indexes_ptr,
    gpu_kv_cache_ptr,
    gpu_stride0,
    gpu_stride1,
    gpu_stride2,
    gpu_stride3,
    gpu_kv_cache_scale_ptr,
    gpu_scale_stride0,
    gpu_scale_stride1,
    gpu_scale_stride2,
    gpu_scale_stride3,
    cpu_kv_cache_ptr,
    cpu_stride0,
    cpu_stride1,
    cpu_stride2,
    cpu_stride3,
    cpu_stride4,
    cpu_kv_cache_scale_ptr,
    cpu_scale_stride0,
    cpu_scale_stride1,
    cpu_scale_stride2,
    cpu_scale_stride3,
    cpu_scale_stride4,
    layer_num,
    head_dim,
    scale_head_dim,
    cpu_k_start_head_index: tl.constexpr,
    cpu_k_head_num: tl.constexpr,
    gpu_k_start_head_index: tl.constexpr,
    gpu_k_head_num: tl.constexpr,
    cpu_v_start_head_index: tl.constexpr,
    cpu_v_head_num: tl.constexpr,
    gpu_v_start_head_index: tl.constexpr,
    gpu_v_head_num: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    block_index_start = tl.program_id(0)
    split_block_num = tl.num_programs(0)
    for block_index in range(block_index_start, copy_block_num, split_block_num):
        token_range = block_index * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        token_mask = token_range < copy_token_num
        gpu_mem_indexes = tl.load(gpu_mem_indexes_ptr + token_range, mask=token_mask).to(tl.int64)
        cpu_mem_indexes = tl.load(cpu_mem_indexes_ptr + token_range, mask=token_mask).to(tl.int64)
        cpu_page_indexes = tl.load(cpu_page_indexes_ptr + token_range, mask=token_mask).to(tl.int64)

        head_dim_range = tl.arange(0, BLOCK_HEAD_DIM)
        head_dim_mask = head_dim_range < head_dim
        scale_head_dim_mask = head_dim_range < scale_head_dim

        for layer_index in range(layer_num):
            move_mask = token_mask[:, None] & head_dim_mask[None, :]
            scale_move_mask = token_mask[:, None] & scale_head_dim_mask[None, :]

            for k_head_index in range(cpu_k_head_num):
                gpu_k_head_index = k_head_index + gpu_k_start_head_index
                cpu_k_head_index = k_head_index + cpu_k_start_head_index

                cpu_ptr = (
                    cpu_kv_cache_ptr
                    + cpu_page_indexes[:, None] * cpu_stride0
                    + layer_index.to(tl.int64) * cpu_stride1
                    + cpu_mem_indexes[:, None] * cpu_stride2
                    + cpu_k_head_index * cpu_stride3
                    + head_dim_range[None, :]
                )
                cpu_data = tl.load(cpu_ptr, mask=move_mask, other=0.0)

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + gpu_mem_indexes[:, None] * gpu_stride1
                    + gpu_k_head_index * gpu_stride2
                    + head_dim_range[None, :]
                )

                tl.store(
                    gpu_ptr,
                    cpu_data,
                    mask=move_mask,
                )
                if HAS_SCALE:
                    cpu_scale_ptr = (
                        cpu_kv_cache_scale_ptr
                        + cpu_page_indexes[:, None] * cpu_scale_stride0
                        + layer_index.to(tl.int64) * cpu_scale_stride1
                        + cpu_mem_indexes[:, None] * cpu_scale_stride2
                        + cpu_k_head_index * cpu_scale_stride3
                        + head_dim_range[None, :]
                    )
                    cpu_scale_data = tl.load(cpu_scale_ptr, mask=scale_move_mask, other=0.0)

                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + gpu_mem_indexes[:, None] * gpu_scale_stride1
                        + gpu_k_head_index * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )

                    tl.store(
                        gpu_scale_ptr,
                        cpu_scale_data,
                        mask=scale_move_mask,
                    )

            for v_head_index in range(cpu_v_head_num):
                gpu_v_head_index = v_head_index + gpu_v_start_head_index
                cpu_v_head_index = v_head_index + cpu_v_start_head_index

                cpu_ptr = (
                    cpu_kv_cache_ptr
                    + cpu_page_indexes[:, None] * cpu_stride0
                    + layer_index.to(tl.int64) * cpu_stride1
                    + cpu_mem_indexes[:, None] * cpu_stride2
                    + cpu_v_head_index * cpu_stride3
                    + head_dim_range[None, :]
                )
                cpu_data = tl.load(cpu_ptr, mask=move_mask, other=0.0)

                gpu_ptr = (
                    gpu_kv_cache_ptr
                    + layer_index.to(tl.int64) * gpu_stride0
                    + gpu_mem_indexes[:, None] * gpu_stride1
                    + gpu_v_head_index * gpu_stride2
                    + head_dim_range[None, :]
                )

                tl.store(
                    gpu_ptr,
                    cpu_data,
                    mask=move_mask,
                )
                if HAS_SCALE:
                    cpu_scale_ptr = (
                        cpu_kv_cache_scale_ptr
                        + cpu_page_indexes[:, None] * cpu_scale_stride0
                        + layer_index.to(tl.int64) * cpu_scale_stride1
                        + cpu_mem_indexes[:, None] * cpu_scale_stride2
                        + cpu_v_head_index * cpu_scale_stride3
                        + head_dim_range[None, :]
                    )
                    cpu_scale_data = tl.load(cpu_scale_ptr, mask=scale_move_mask, other=0.0)

                    gpu_scale_ptr = (
                        gpu_kv_cache_scale_ptr
                        + layer_index.to(tl.int64) * gpu_scale_stride0
                        + gpu_mem_indexes[:, None] * gpu_scale_stride1
                        + gpu_v_head_index * gpu_scale_stride2
                        + head_dim_range[None, :]
                    )

                    tl.store(
                        gpu_scale_ptr,
                        cpu_scale_data,
                        mask=scale_move_mask,
                    )

    return


@torch.no_grad()
def load_cpu_kv_to_gpu(
    gpu_mem_indexes: torch.Tensor,
    gpu_kv_cache: torch.Tensor,
    gpu_kv_cache_scale: Optional[torch.Tensor],
    cpu_kv_cache: torch.Tensor,
    cpu_kv_cache_scale: Optional[torch.Tensor],
    page_indexes: torch.Tensor,
    tp_index: int,
    tp_world_size: int,
    grid_num: int,
    _cache_data={},
):
    """
    this function is used to offload GPU KV cache to CPU KV cache.
    Args:
        gpu_mem_indexes: (token_num,)
        gpu_kv_cache: (layer_num, all_token_num, head_num, head_dim)
        cpu_kv_cache: (all_page_num, layer_num, token_block_size, head_num, head_dim)
        page_indexes: (page_num,)
    """
    token_block_size = cpu_kv_cache.shape[2]
    cpu_page_num = page_indexes.shape[0]
    cpu_page_all_token_num = cpu_page_num * token_block_size
    assert gpu_mem_indexes.shape[0] <= cpu_page_all_token_num
    move_token_num = gpu_mem_indexes.shape[0]

    cpu_page_indexes = page_indexes.view((cpu_page_num, 1)).tile((1, token_block_size)).view(-1)
    cpu_mem_indexes = torch.arange(0, cpu_page_all_token_num, device="cuda", dtype=torch.int32) % token_block_size
    cpu_page_indexes = cpu_page_indexes[-move_token_num:]
    cpu_mem_indexes = cpu_mem_indexes[-move_token_num:]

    gpu_heads = gpu_kv_cache.shape[2]
    gpu_head_dim = gpu_kv_cache.shape[3]
    cpu_heads = cpu_kv_cache.shape[3]
    cpu_head_dim = cpu_kv_cache.shape[4]
    assert gpu_head_dim == cpu_head_dim
    head_dim = gpu_head_dim
    scale_size = (tp_world_size * gpu_heads) // cpu_heads

    # 计算需要拷贝的 head 索引的对应关系
    if (gpu_heads, cpu_heads, tp_index, tp_world_size) in _cache_data:
        head_info_tuple = _cache_data[(gpu_heads, cpu_heads, tp_index, tp_world_size)]
    else:
        if cpu_heads > 1:
            assert (tp_world_size * gpu_heads) % cpu_heads == 0
            assert cpu_heads % 2 == 0

            cpu_heads_index = (
                torch.arange(0, cpu_heads, device="cpu", dtype=torch.int32)
                .view(cpu_heads, 1)
                .tile((1, scale_size))
                .view(2, tp_world_size, -1)
            )
            # k
            k_cpu_heads_index = cpu_heads_index[0][tp_index]
            # v
            v_cpu_heads_index = cpu_heads_index[1][tp_index]

            cpu_heads_index = torch.cat([k_cpu_heads_index, v_cpu_heads_index], dim=0).view(2, -1).numpy()
            gpu_heads_index = torch.arange(0, gpu_heads, device="cpu", dtype=torch.int32).view(2, -1).numpy()

            cpu_k_start_head_index = int(cpu_heads_index[0, 0])
            cpu_k_head_num = len(cpu_heads_index[0])
            gpu_k_start_head_index = int(gpu_heads_index[0, 0])
            gpu_k_head_num = len(gpu_heads_index[0])
            assert cpu_k_head_num == gpu_k_head_num
            cpu_v_start_head_index = int(cpu_heads_index[1, 0])
            cpu_v_head_num = len(cpu_heads_index[1])
            gpu_v_start_head_index = int(gpu_heads_index[1, 0])
            gpu_v_head_num = len(gpu_heads_index[1])
            assert cpu_v_head_num == gpu_v_head_num

            head_info_tuple = (
                cpu_k_start_head_index,
                cpu_k_head_num,
                gpu_k_start_head_index,
                gpu_k_head_num,
                cpu_v_start_head_index,
                cpu_v_head_num,
                gpu_v_start_head_index,
                gpu_v_head_num,
            )

        else:
            assert gpu_heads == 1
            assert cpu_heads == 1

            cpu_k_start_head_index = 0
            cpu_k_head_num = 1
            gpu_k_start_head_index = 0
            gpu_k_head_num = 1
            cpu_v_start_head_index = 0
            cpu_v_head_num = 0
            gpu_v_start_head_index = 0
            gpu_v_head_num = 0
            head_info_tuple = (
                cpu_k_start_head_index,
                cpu_k_head_num,
                gpu_k_start_head_index,
                gpu_k_head_num,
                cpu_v_start_head_index,
                cpu_v_head_num,
                gpu_v_start_head_index,
                gpu_v_head_num,
            )

        _cache_data[(gpu_heads, cpu_heads, tp_index, tp_world_size)] = head_info_tuple

    (
        cpu_k_start_head_index,
        cpu_k_head_num,
        gpu_k_start_head_index,
        gpu_k_head_num,
        cpu_v_start_head_index,
        cpu_v_head_num,
        gpu_v_start_head_index,
        gpu_v_head_num,
    ) = head_info_tuple

    TOKEN_BLOCK = 128

    grid = (grid_num,)
    num_warps = 4

    HAS_SCALE = gpu_kv_cache_scale is not None and cpu_kv_cache_scale is not None
    if HAS_SCALE:
        scale_head_dim = gpu_kv_cache_scale.shape[-1]
        gpu_scale_stride = gpu_kv_cache_scale.stride()
        cpu_scale_stride = cpu_kv_cache_scale.stride()
    else:
        scale_head_dim = 0
        gpu_scale_stride = [0 for _ in range(10)]
        cpu_scale_stride = [0 for _ in range(10)]

    _load_cpu_cache_to_gpu[grid](
        gpu_mem_indexes_ptr=gpu_mem_indexes,
        copy_token_num=move_token_num,
        copy_block_num=triton.cdiv(move_token_num, TOKEN_BLOCK),
        cpu_mem_indexes_ptr=cpu_mem_indexes,
        cpu_page_indexes_ptr=cpu_page_indexes,
        gpu_kv_cache_ptr=gpu_kv_cache,
        gpu_stride0=gpu_kv_cache.stride(0),
        gpu_stride1=gpu_kv_cache.stride(1),
        gpu_stride2=gpu_kv_cache.stride(2),
        gpu_stride3=gpu_kv_cache.stride(3),
        gpu_kv_cache_scale_ptr=gpu_kv_cache_scale,
        gpu_scale_stride0=gpu_scale_stride[0],
        gpu_scale_stride1=gpu_scale_stride[1],
        gpu_scale_stride2=gpu_scale_stride[2],
        gpu_scale_stride3=gpu_scale_stride[3],
        cpu_kv_cache_ptr=cpu_kv_cache,
        cpu_stride0=cpu_kv_cache.stride(0),
        cpu_stride1=cpu_kv_cache.stride(1),
        cpu_stride2=cpu_kv_cache.stride(2),
        cpu_stride3=cpu_kv_cache.stride(3),
        cpu_stride4=cpu_kv_cache.stride(4),
        cpu_kv_cache_scale_ptr=cpu_kv_cache_scale,
        cpu_scale_stride0=cpu_scale_stride[0],
        cpu_scale_stride1=cpu_scale_stride[1],
        cpu_scale_stride2=cpu_scale_stride[2],
        cpu_scale_stride3=cpu_scale_stride[3],
        cpu_scale_stride4=cpu_scale_stride[4],
        layer_num=gpu_kv_cache.shape[0],
        head_dim=head_dim,
        scale_head_dim=scale_head_dim,
        cpu_k_start_head_index=cpu_k_start_head_index,
        cpu_k_head_num=cpu_k_head_num,
        gpu_k_start_head_index=gpu_k_start_head_index,
        gpu_k_head_num=gpu_k_head_num,
        cpu_v_start_head_index=cpu_v_start_head_index,
        cpu_v_head_num=cpu_v_head_num,
        gpu_v_start_head_index=gpu_v_start_head_index,
        gpu_v_head_num=gpu_v_head_num,
        BLOCK_HEAD_DIM=triton.next_power_of_2(head_dim),
        TOKEN_BLOCK=TOKEN_BLOCK,
        HAS_SCALE=HAS_SCALE,
        num_warps=num_warps,
        num_stages=1,
    )
    return

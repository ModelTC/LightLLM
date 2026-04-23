import torch
import triton
import triton.language as tl
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig


@triton.jit
def _copy_kv_buffer_to_cpu_cache(
    page_num,
    mem_indexes_ptr,  # [move_token_num]
    page_indexes_ptr,  # [page_num],
    page_readies_ptr,  # [page_num],
    gpu_full_att_ptr,  # [token_size, full_att_layer_num, xdim]
    cpu_kv_conv_ptr,  # [token_size, linear_att_layer_num, xxdim]
    cpu_kv_ssm_ptr,  # [token_size, linear_att_layer_num, xxxdim]
    cpu_cache_tensor_ptr,  # [page_num, tp_world_size, xxxxdim]
    gpu_full_att_stride_s,
    gpu_full_att_stride_l,
    gpu_full_att_stride_d,
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_l,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_l,
    cpu_kv_ssm_stride_d,
    cpu_cache_tensor_stride_page,
    cpu_cache_tensor_stride_t,
    cpu_cache_tensor_stride_d,
    gpu_full_att_tail_dim,
    cpu_kv_conv_tail_dim,
    cpu_kv_ssm_tail_dim,
    conv_bytes_per_layer,
    ssm_bytes_per_layer,
    tp_rank,
    full_att_layer_num,
    linear_layer_num,
    big_page_token_num,
    BLOCK: tl.constexpr,
):
    block_index_start = tl.program_id(0)
    grid_num = tl.num_programs(0)
    gpu_full_att_stride_s = tl.cast(gpu_full_att_stride_s, tl.int64)
    gpu_full_att_stride_l = tl.cast(gpu_full_att_stride_l, tl.int64)
    cpu_kv_conv_stride_s = tl.cast(cpu_kv_conv_stride_s, tl.int64)
    cpu_kv_conv_stride_l = tl.cast(cpu_kv_conv_stride_l, tl.int64)
    cpu_kv_ssm_stride_s = tl.cast(cpu_kv_ssm_stride_s, tl.int64)
    cpu_kv_ssm_stride_l = tl.cast(cpu_kv_ssm_stride_l, tl.int64)
    cpu_cache_tensor_stride_page = tl.cast(cpu_cache_tensor_stride_page, tl.int64)
    cpu_cache_tensor_stride_t = tl.cast(cpu_cache_tensor_stride_t, tl.int64)

    for block_index in range(block_index_start, page_num, grid_num):
        cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)
        run_flag = 1
        if cpu_page_index == -1:
            run_flag = 0
        ready_state = tl.load(page_readies_ptr + block_index)
        if ready_state:
            run_flag = 0

        mem_start_ptr = mem_indexes_ptr + big_page_token_num * block_index
        full_att_big_page_bytes = full_att_layer_num * gpu_full_att_tail_dim * big_page_token_num
        full_att_dest_start = 0
        for i in range(tl.cdiv(full_att_big_page_bytes, BLOCK) * run_flag):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < full_att_big_page_bytes
            mem_offs = gpu_start_i // (full_att_layer_num * gpu_full_att_tail_dim)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-11111111)
            layer_index = (gpu_start_i // (gpu_full_att_tail_dim)) % full_att_layer_num
            dim_index = gpu_start_i % gpu_full_att_tail_dim
            gpu_full_att_data = tl.load(
                gpu_full_att_ptr
                + mem_index * gpu_full_att_stride_s
                + layer_index * gpu_full_att_stride_l
                + dim_index * gpu_full_att_stride_d,
                mask=mask,
                other=0,
            )
            dest_cpu_cache_ptr = (
                cpu_cache_tensor_ptr
                + cpu_page_index * cpu_cache_tensor_stride_page
                + tp_rank * cpu_cache_tensor_stride_t
                + (full_att_dest_start + gpu_start_i) * cpu_cache_tensor_stride_d
            )
            tl.store(dest_cpu_cache_ptr, gpu_full_att_data, mask=mask)

        linear_att_conv_big_page_bytes = linear_layer_num * conv_bytes_per_layer
        linear_att_dest_start = full_att_big_page_bytes
        for i in range(tl.cdiv(linear_att_conv_big_page_bytes, BLOCK) * run_flag):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < linear_att_conv_big_page_bytes
            mem_offs = gpu_start_i // (linear_layer_num * cpu_kv_conv_tail_dim)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-11111111)
            layer_index = (gpu_start_i // (cpu_kv_conv_tail_dim)) % linear_layer_num
            dim_index = gpu_start_i % cpu_kv_conv_tail_dim
            cpu_kv_conv_data = tl.load(
                cpu_kv_conv_ptr
                + mem_index * cpu_kv_conv_stride_s
                + layer_index * cpu_kv_conv_stride_l
                + dim_index * cpu_kv_conv_stride_d,
                mask=mask,
                other=0,
            )
            dest_cpu_cache_ptr = (
                cpu_cache_tensor_ptr
                + cpu_page_index * cpu_cache_tensor_stride_page
                + tp_rank * cpu_cache_tensor_stride_t
                + (linear_att_dest_start + gpu_start_i) * cpu_cache_tensor_stride_d
            )
            tl.store(dest_cpu_cache_ptr, cpu_kv_conv_data, mask=mask)

        linear_att_ssm_big_page_bytes = linear_layer_num * ssm_bytes_per_layer
        linear_att_dest_start = full_att_big_page_bytes + linear_att_conv_big_page_bytes
        for i in range(tl.cdiv(linear_att_ssm_big_page_bytes, BLOCK) * run_flag):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < linear_att_ssm_big_page_bytes
            mem_offs = gpu_start_i // (linear_layer_num * cpu_kv_ssm_tail_dim)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-11111111)
            layer_index = (gpu_start_i // (cpu_kv_ssm_tail_dim)) % linear_layer_num
            dim_index = gpu_start_i % cpu_kv_ssm_tail_dim
            cpu_kv_ssm_data = tl.load(
                cpu_kv_ssm_ptr
                + mem_index * cpu_kv_ssm_stride_s
                + layer_index * cpu_kv_ssm_stride_l
                + dim_index * cpu_kv_ssm_stride_d,
                mask=mask,
                other=0,
            )
            dest_cpu_cache_ptr = (
                cpu_cache_tensor_ptr
                + cpu_page_index * cpu_cache_tensor_stride_page
                + tp_rank * cpu_cache_tensor_stride_t
                + (linear_att_dest_start + gpu_start_i) * cpu_cache_tensor_stride_d
            )
            tl.store(dest_cpu_cache_ptr, cpu_kv_ssm_data, mask=mask)

    return


def copy_kv_buffer_to_cpu_cache(
    mem_indexes: torch.Tensor,
    page_indexes: torch.Tensor,
    page_readies: torch.Tensor,
    gpu_full_att_kv_state: torch.Tensor,  # [layer_num, s, head_num, head_dim]
    cpu_kv_conv_state: torch.Tensor,  # [layer_num, s, dim]
    cpu_kv_ssm_state: torch.Tensor,  # [layer_num, s, xdim]
    cpu_cache_tensor: torch.Tensor,  # [page_num, 1, 1, tp_world_size, xxdim]
    tp_rank: int,
    tp_world_size: int,
    big_page_token_num: int,
    linear_config: LinearAttCacheConfig,
    grid_num: int = 16,
):
    BLOCK = 4096
    gpu_full_att_kv_state = gpu_full_att_kv_state.view(
        gpu_full_att_kv_state.shape[0], gpu_full_att_kv_state.shape[1], -1
    ).view(dtype=torch.uint8)
    gpu_full_att_kv_state = gpu_full_att_kv_state.permute(1, 0, 2)  # [s, layer_num, xxxdim]
    cpu_kv_conv_state = cpu_kv_conv_state.permute(1, 0, 2)  # [s, layer_num, xxdim]
    cpu_kv_ssm_state = cpu_kv_ssm_state.permute(1, 0, 2)  # [s, layer_num, xdim]

    gpu_full_att_tail_dim = gpu_full_att_kv_state.shape[-1]
    cpu_kv_conv_tail_dim = cpu_kv_conv_state.shape[-1]
    cpu_kv_ssm_tail_dim = cpu_kv_ssm_state.shape[-1]

    linear_layer_num = cpu_kv_conv_state.shape[1]
    full_att_layer_num = gpu_full_att_kv_state.shape[1]

    cpu_dim_bytes = linear_config.get_cpu_cache_big_page_bytes()
    assert cpu_dim_bytes == cpu_cache_tensor.shape[-1]
    cpu_cache_tensor = cpu_cache_tensor.view(
        cpu_cache_tensor.shape[0], cpu_cache_tensor.shape[-2], cpu_cache_tensor.shape[-1]
    )
    assert cpu_cache_tensor.shape[1] == tp_world_size

    grid = (grid_num,)
    _copy_kv_buffer_to_cpu_cache[grid](
        page_num=len(page_indexes),
        mem_indexes_ptr=mem_indexes,
        page_indexes_ptr=page_indexes,
        page_readies_ptr=page_readies,
        gpu_full_att_ptr=gpu_full_att_kv_state,
        cpu_kv_conv_ptr=cpu_kv_conv_state,
        cpu_kv_ssm_ptr=cpu_kv_ssm_state,
        cpu_cache_tensor_ptr=cpu_cache_tensor,
        gpu_full_att_stride_s=gpu_full_att_kv_state.stride(0),
        gpu_full_att_stride_l=gpu_full_att_kv_state.stride(1),
        gpu_full_att_stride_d=gpu_full_att_kv_state.stride(2),
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_l=cpu_kv_conv_state.stride(1),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(2),
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_l=cpu_kv_ssm_state.stride(1),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(2),
        cpu_cache_tensor_stride_page=cpu_cache_tensor.stride(0),
        cpu_cache_tensor_stride_t=cpu_cache_tensor.stride(1),
        cpu_cache_tensor_stride_d=cpu_cache_tensor.stride(2),
        gpu_full_att_tail_dim=gpu_full_att_tail_dim,
        cpu_kv_conv_tail_dim=cpu_kv_conv_tail_dim,
        cpu_kv_ssm_tail_dim=cpu_kv_ssm_tail_dim,
        conv_bytes_per_layer=linear_config.get_conv_state_bytes(),
        ssm_bytes_per_layer=linear_config.get_ssm_state_bytes(),
        tp_rank=tp_rank,
        full_att_layer_num=full_att_layer_num,
        linear_layer_num=linear_layer_num,
        big_page_token_num=big_page_token_num,
        BLOCK=BLOCK,
    )


@triton.jit
def _copy_cpu_cache_to_kv_buffer(
    page_num,
    mem_indexes_ptr,  # [move_token_num]
    page_indexes_ptr,  # [page_num],
    gpu_full_att_ptr,  # [token_size, full_att_layer_num, xdim]
    cpu_kv_conv_ptr,  # [token_size, linear_att_layer_num, xxdim]
    cpu_kv_ssm_ptr,  # [token_size, linear_att_layer_num, xxxdim]
    cpu_cache_tensor_ptr,  # [page_num, tp_world_size, xxxxdim]
    gpu_full_att_stride_s,
    gpu_full_att_stride_l,
    gpu_full_att_stride_d,
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_l,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_l,
    cpu_kv_ssm_stride_d,
    cpu_cache_tensor_stride_page,
    cpu_cache_tensor_stride_t,
    cpu_cache_tensor_stride_d,
    gpu_full_att_tail_dim,
    cpu_kv_conv_tail_dim,
    cpu_kv_ssm_tail_dim,
    conv_bytes_per_layer,
    ssm_bytes_per_layer,
    tp_rank,
    full_att_layer_num,
    linear_layer_num,
    big_page_token_num,
    BLOCK: tl.constexpr,
):
    block_index_start = tl.program_id(0)
    grid_num = tl.num_programs(0)
    gpu_full_att_stride_s = tl.cast(gpu_full_att_stride_s, tl.int64)
    gpu_full_att_stride_l = tl.cast(gpu_full_att_stride_l, tl.int64)
    cpu_kv_conv_stride_s = tl.cast(cpu_kv_conv_stride_s, tl.int64)
    cpu_kv_conv_stride_l = tl.cast(cpu_kv_conv_stride_l, tl.int64)
    cpu_kv_ssm_stride_s = tl.cast(cpu_kv_ssm_stride_s, tl.int64)
    cpu_kv_ssm_stride_l = tl.cast(cpu_kv_ssm_stride_l, tl.int64)
    cpu_cache_tensor_stride_page = tl.cast(cpu_cache_tensor_stride_page, tl.int64)
    cpu_cache_tensor_stride_t = tl.cast(cpu_cache_tensor_stride_t, tl.int64)

    for block_index in range(block_index_start, page_num, grid_num):
        cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)
        mem_start_ptr = mem_indexes_ptr + big_page_token_num * block_index
        full_att_big_page_bytes = full_att_layer_num * gpu_full_att_tail_dim * big_page_token_num
        full_att_dest_start = 0
        for i in range(tl.cdiv(full_att_big_page_bytes, BLOCK)):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < full_att_big_page_bytes
            src_cpu_cache_ptr = (
                cpu_cache_tensor_ptr
                + cpu_page_index * cpu_cache_tensor_stride_page
                + tp_rank * cpu_cache_tensor_stride_t
                + (full_att_dest_start + gpu_start_i) * cpu_cache_tensor_stride_d
            )
            full_att_data = tl.load(src_cpu_cache_ptr, mask=mask, other=0)

            mem_offs = gpu_start_i // (full_att_layer_num * gpu_full_att_tail_dim)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-11111111)
            layer_index = (gpu_start_i // (gpu_full_att_tail_dim)) % full_att_layer_num
            dim_index = gpu_start_i % gpu_full_att_tail_dim
            tl.store(
                gpu_full_att_ptr
                + mem_index * gpu_full_att_stride_s
                + layer_index * gpu_full_att_stride_l
                + dim_index * gpu_full_att_stride_d,
                full_att_data,
                mask=mask,
            )

        linear_att_conv_big_page_bytes = linear_layer_num * conv_bytes_per_layer
        linear_att_dest_start = full_att_big_page_bytes
        for i in range(tl.cdiv(linear_att_conv_big_page_bytes, BLOCK)):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < linear_att_conv_big_page_bytes
            src_cpu_cache_ptr = (
                cpu_cache_tensor_ptr
                + cpu_page_index * cpu_cache_tensor_stride_page
                + tp_rank * cpu_cache_tensor_stride_t
                + (linear_att_dest_start + gpu_start_i) * cpu_cache_tensor_stride_d
            )
            conv_data = tl.load(src_cpu_cache_ptr, mask=mask, other=0)

            mem_offs = gpu_start_i // (linear_layer_num * cpu_kv_conv_tail_dim)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-11111111)
            layer_index = (gpu_start_i // (cpu_kv_conv_tail_dim)) % linear_layer_num
            dim_index = gpu_start_i % cpu_kv_conv_tail_dim
            tl.store(
                cpu_kv_conv_ptr
                + mem_index * cpu_kv_conv_stride_s
                + layer_index * cpu_kv_conv_stride_l
                + dim_index * cpu_kv_conv_stride_d,
                conv_data,
                mask=mask,
            )

        linear_att_ssm_big_page_bytes = linear_layer_num * ssm_bytes_per_layer
        linear_att_dest_start = full_att_big_page_bytes + linear_att_conv_big_page_bytes
        for i in range(tl.cdiv(linear_att_ssm_big_page_bytes, BLOCK)):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < linear_att_ssm_big_page_bytes

            src_cpu_cache_ptr = (
                cpu_cache_tensor_ptr
                + cpu_page_index * cpu_cache_tensor_stride_page
                + tp_rank * cpu_cache_tensor_stride_t
                + (linear_att_dest_start + gpu_start_i) * cpu_cache_tensor_stride_d
            )

            ssm_data = tl.load(src_cpu_cache_ptr, mask=mask, other=0)

            mem_offs = gpu_start_i // (linear_layer_num * cpu_kv_ssm_tail_dim)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-11111111)
            layer_index = (gpu_start_i // (cpu_kv_ssm_tail_dim)) % linear_layer_num
            dim_index = gpu_start_i % cpu_kv_ssm_tail_dim
            tl.store(
                cpu_kv_ssm_ptr
                + mem_index * cpu_kv_ssm_stride_s
                + layer_index * cpu_kv_ssm_stride_l
                + dim_index * cpu_kv_ssm_stride_d,
                ssm_data,
                mask=mask,
            )
    return


def copy_cpu_cache_to_kv_buffer(
    mem_indexes: torch.Tensor,
    page_indexes: torch.Tensor,
    gpu_full_att_kv_state: torch.Tensor,  # [layer_num, s, head_num, head_dim]
    cpu_kv_conv_state: torch.Tensor,  # [layer_num, s, dim]
    cpu_kv_ssm_state: torch.Tensor,  # [layer_num, s, xdim]
    cpu_cache_tensor: torch.Tensor,  # [page_num, 1, 1, tp_world_size, xxdim]
    tp_rank: int,
    tp_world_size: int,
    big_page_token_num: int,
    linear_config: LinearAttCacheConfig,
    grid_num: int = 16,
):
    assert len(mem_indexes) % len(page_indexes) == 0
    BLOCK = 4096
    gpu_full_att_kv_state = gpu_full_att_kv_state.view(
        gpu_full_att_kv_state.shape[0], gpu_full_att_kv_state.shape[1], -1
    ).view(dtype=torch.uint8)
    gpu_full_att_kv_state = gpu_full_att_kv_state.permute(1, 0, 2)  # [s, layer_num, xxxdim]
    cpu_kv_conv_state = cpu_kv_conv_state.permute(1, 0, 2)  # [s, layer_num, xxdim]
    cpu_kv_ssm_state = cpu_kv_ssm_state.permute(1, 0, 2)  # [s, layer_num, xdim]

    gpu_full_att_tail_dim = gpu_full_att_kv_state.shape[-1]
    cpu_kv_conv_tail_dim = cpu_kv_conv_state.shape[-1]
    cpu_kv_ssm_tail_dim = cpu_kv_ssm_state.shape[-1]

    linear_layer_num = cpu_kv_conv_state.shape[1]
    full_att_layer_num = gpu_full_att_kv_state.shape[1]

    cpu_dim_bytes = linear_config.get_cpu_cache_big_page_bytes()
    assert cpu_dim_bytes == cpu_cache_tensor.shape[-1]
    cpu_cache_tensor = cpu_cache_tensor.view(
        cpu_cache_tensor.shape[0], cpu_cache_tensor.shape[-2], cpu_cache_tensor.shape[-1]
    )
    assert cpu_cache_tensor.shape[1] == tp_world_size

    grid = (grid_num,)
    _copy_cpu_cache_to_kv_buffer[grid](
        page_num=len(page_indexes),
        mem_indexes_ptr=mem_indexes,
        page_indexes_ptr=page_indexes,
        gpu_full_att_ptr=gpu_full_att_kv_state,
        cpu_kv_conv_ptr=cpu_kv_conv_state,
        cpu_kv_ssm_ptr=cpu_kv_ssm_state,
        cpu_cache_tensor_ptr=cpu_cache_tensor,
        gpu_full_att_stride_s=gpu_full_att_kv_state.stride(0),
        gpu_full_att_stride_l=gpu_full_att_kv_state.stride(1),
        gpu_full_att_stride_d=gpu_full_att_kv_state.stride(2),
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_l=cpu_kv_conv_state.stride(1),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(2),
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_l=cpu_kv_ssm_state.stride(1),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(2),
        cpu_cache_tensor_stride_page=cpu_cache_tensor.stride(0),
        cpu_cache_tensor_stride_t=cpu_cache_tensor.stride(1),
        cpu_cache_tensor_stride_d=cpu_cache_tensor.stride(2),
        gpu_full_att_tail_dim=gpu_full_att_tail_dim,
        cpu_kv_conv_tail_dim=cpu_kv_conv_tail_dim,
        cpu_kv_ssm_tail_dim=cpu_kv_ssm_tail_dim,
        conv_bytes_per_layer=linear_config.get_conv_state_bytes(),
        ssm_bytes_per_layer=linear_config.get_ssm_state_bytes(),
        tp_rank=tp_rank,
        full_att_layer_num=full_att_layer_num,
        linear_layer_num=linear_layer_num,
        big_page_token_num=big_page_token_num,
        BLOCK=BLOCK,
    )

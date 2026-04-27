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
    big_page_buffer_ids,  # [page_num]
    cpu_cache_full_att,  # [all_page_num, head, xdim]
    cpu_cache_full_att_stride_p,
    cpu_cache_full_att_stride_h,
    cpu_cache_full_att_stride_d,
    cpu_cache_conv,  # [all_page_num, tp_world_size, xxdim]
    cpu_cache_conv_stride_p,
    cpu_cache_conv_stride_t,
    cpu_cache_conv_stride_d,
    cpu_cache_ssm,  # [all_page_num, tp_world_size, xxxdim]
    cpu_cache_ssm_stride_p,
    cpu_cache_ssm_stride_t,
    cpu_cache_ssm_stride_d,
    gpu_kv_full_att_state,  # [token_size, full_att_layer_num, xdim]
    gpu_kv_full_att_stride_s,
    gpu_kv_full_att_stride_l,
    gpu_kv_full_att_stride_d,
    cpu_kv_conv_state,  # [buffer_count, xxxxxdim]
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_state,  # [buffer_count, xxxxxxxdim]
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_d,
    gpu_full_att_tail_dim,
    cpu_kv_conv_tail_dim,
    cpu_kv_ssm_tail_dim,
    tp_rank,
    full_att_layer_num,
    big_page_token_num,
    head_scale_size,
    BLOCK: tl.constexpr,
):
    block_index_start = tl.program_id(0)
    grid_num = tl.num_programs(0)
    # 将 所有stride 切成 tl.int64
    cpu_cache_full_att_stride_p = tl.cast(cpu_cache_full_att_stride_p, tl.int64)
    cpu_cache_full_att_stride_h = tl.cast(cpu_cache_full_att_stride_h, tl.int64)
    cpu_cache_full_att_stride_d = tl.cast(cpu_cache_full_att_stride_d, tl.int64)
    cpu_cache_conv_stride_p = tl.cast(cpu_cache_conv_stride_p, tl.int64)
    cpu_cache_conv_stride_t = tl.cast(cpu_cache_conv_stride_t, tl.int64)
    cpu_cache_conv_stride_d = tl.cast(cpu_cache_conv_stride_d, tl.int64)
    cpu_cache_ssm_stride_p = tl.cast(cpu_cache_ssm_stride_p, tl.int64)
    cpu_cache_ssm_stride_t = tl.cast(cpu_cache_ssm_stride_t, tl.int64)
    cpu_cache_ssm_stride_d = tl.cast(cpu_cache_ssm_stride_d, tl.int64)
    gpu_kv_full_att_stride_s = tl.cast(gpu_kv_full_att_stride_s, tl.int64)
    gpu_kv_full_att_stride_l = tl.cast(gpu_kv_full_att_stride_l, tl.int64)
    gpu_kv_full_att_stride_d = tl.cast(gpu_kv_full_att_stride_d, tl.int64)
    cpu_kv_conv_stride_s = tl.cast(cpu_kv_conv_stride_s, tl.int64)
    cpu_kv_conv_stride_d = tl.cast(cpu_kv_conv_stride_d, tl.int64)
    cpu_kv_ssm_stride_s = tl.cast(cpu_kv_ssm_stride_s, tl.int64)
    cpu_kv_ssm_stride_d = tl.cast(cpu_kv_ssm_stride_d, tl.int64)

    for block_index in range(block_index_start, page_num, grid_num):
        cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)
        run_flag = 1
        if cpu_page_index == -1:
            run_flag = 0
        ready_state = tl.load(page_readies_ptr + block_index)
        if ready_state:
            run_flag = 0
        if tp_rank % head_scale_size == 0:
            head_flag = 1
        else:
            head_flag = 0

        mem_start_ptr = mem_indexes_ptr + big_page_token_num * block_index
        for i in range(tl.cdiv(gpu_full_att_tail_dim, BLOCK) * run_flag * head_flag):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < gpu_full_att_tail_dim
            per_token_size = gpu_full_att_tail_dim // big_page_token_num
            per_layer_size = per_token_size // full_att_layer_num
            mem_offs = gpu_start_i // (per_token_size)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-11111111)
            layer_index = (gpu_start_i // (per_layer_size)) % full_att_layer_num
            dim_index = gpu_start_i % per_layer_size
            gpu_full_att_data = tl.load(
                gpu_kv_full_att_state
                + mem_index * gpu_kv_full_att_stride_s
                + layer_index * gpu_kv_full_att_stride_l
                + dim_index * gpu_kv_full_att_stride_d,
                mask=mask,
                other=0,
            )
            dest_cpu_cache_full_att_ptr = (
                cpu_cache_full_att
                + cpu_page_index * cpu_cache_full_att_stride_p
                + (tp_rank // head_scale_size) * cpu_cache_full_att_stride_h
                + gpu_start_i
            )
            tl.store(dest_cpu_cache_full_att_ptr, gpu_full_att_data, mask=mask)

        big_page_idx = tl.load(big_page_buffer_ids + block_index)

        for i in range(tl.cdiv(cpu_kv_conv_tail_dim, BLOCK) * run_flag):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < cpu_kv_conv_tail_dim
            cpu_kv_conv_data = tl.load(
                cpu_kv_conv_state + big_page_idx * cpu_kv_conv_stride_s + gpu_start_i,
                mask=mask,
                other=0,
            )
            dest_cpu_cache_conv_ptr = (
                cpu_cache_conv
                + cpu_page_index * cpu_cache_conv_stride_p
                + tp_rank * cpu_cache_conv_stride_t
                + gpu_start_i
            )
            tl.store(dest_cpu_cache_conv_ptr, cpu_kv_conv_data, mask=mask)

        for i in range(tl.cdiv(cpu_kv_ssm_tail_dim, BLOCK) * run_flag):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < cpu_kv_ssm_tail_dim

            cpu_kv_ssm_data = tl.load(
                cpu_kv_ssm_state + big_page_idx * cpu_kv_ssm_stride_s + gpu_start_i,
                mask=mask,
                other=0,
            )
            dest_cpu_cache_ssm_ptr = (
                cpu_cache_ssm + cpu_page_index * cpu_cache_ssm_stride_p + tp_rank * cpu_cache_ssm_stride_t + gpu_start_i
            )
            tl.store(dest_cpu_cache_ssm_ptr, cpu_kv_ssm_data, mask=mask)

    return


def copy_kv_buffer_to_cpu_cache(
    mem_indexes: torch.Tensor,
    page_indexes: torch.Tensor,
    page_readies: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
    gpu_kv_full_att_state: torch.Tensor,  # [full_att_layer_num, s, head_num, head_dim]
    cpu_kv_conv_state: torch.Tensor,  # [s, linear_layer_num, dim]
    cpu_kv_ssm_state: torch.Tensor,  # [s, linear_layer_num, xdim]
    cpu_cache_tensor: torch.Tensor,  # [page_num, 1, 1, 1, xxdim]
    tp_rank: int,
    tp_world_size: int,
    big_page_token_num: int,
    linear_config: LinearAttCacheConfig,
    grid_num: int = 16,
):
    assert len(page_indexes) == len(page_readies) == len(big_page_buffer_ids)
    assert len(mem_indexes) % len(page_indexes) == 0

    BLOCK = 4096
    if linear_config.full_att_all_num_kv_heads % tp_world_size == 0:
        # tp world size 不比 kv 的 head 多时
        head_scale_size = 1
    else:
        head_scale_size = tp_world_size // linear_config.full_att_all_num_kv_heads

    cpu_page_num = cpu_cache_tensor.shape[0]
    cpu_cache_tensor = cpu_cache_tensor.view(cpu_page_num, -1).view(dtype=torch.uint8)
    a = linear_config.get_cpu_cache_full_att_bytes()
    b = linear_config.get_cpu_cache_conv_bytes()
    c = linear_config.get_cpu_cache_ssm_bytes()

    if head_scale_size == 1:
        cpu_cache_full_att = cpu_cache_tensor[:, 0:a].view(cpu_page_num, tp_world_size, -1)
    else:
        cpu_cache_full_att = cpu_cache_tensor[:, 0:a].view(cpu_page_num, linear_config.full_att_all_num_kv_heads, -1)

    cpu_cache_conv = cpu_cache_tensor[:, a : (a + b)].view(cpu_page_num, tp_world_size, -1)
    cpu_cache_ssm = cpu_cache_tensor[:, (a + b) : (a + b + c)].view(cpu_page_num, tp_world_size, -1)

    gpu_kv_full_att_state = gpu_kv_full_att_state.view(
        gpu_kv_full_att_state.shape[0], gpu_kv_full_att_state.shape[1], -1
    ).view(dtype=torch.uint8)
    gpu_kv_full_att_state = gpu_kv_full_att_state.permute(1, 0, 2)  # [s, layer_num, xxdim]
    cpu_kv_conv_state = cpu_kv_conv_state.view(cpu_kv_conv_state.shape[0], -1).view(dtype=torch.uint8)
    cpu_kv_ssm_state = cpu_kv_ssm_state.view(cpu_kv_ssm_state.shape[0], -1).view(dtype=torch.uint8)

    gpu_full_att_tail_dim = gpu_kv_full_att_state.shape[-1] * gpu_kv_full_att_state.shape[-2] * big_page_token_num
    cpu_kv_conv_tail_dim = cpu_kv_conv_state.shape[-1]
    cpu_kv_ssm_tail_dim = cpu_kv_ssm_state.shape[-1]
    full_att_layer_num = gpu_kv_full_att_state.shape[-2]

    assert (
        full_att_layer_num
        == (linear_config.all_layer_num // linear_config.full_attention_interval)
        == (linear_config.all_layer_num - linear_config.linear_layer_num)
    )
    assert gpu_full_att_tail_dim == cpu_cache_full_att.shape[-1]
    assert cpu_cache_conv.shape[-1] == cpu_kv_conv_state.shape[-1]
    assert cpu_cache_ssm.shape[-1] == cpu_kv_ssm_state.shape[-1]
    assert (
        gpu_full_att_tail_dim % big_page_token_num == 0
        and (gpu_full_att_tail_dim // big_page_token_num) % full_att_layer_num == 0
    )
    assert (tp_rank // head_scale_size) < linear_config.full_att_all_num_kv_heads

    grid = (grid_num,)
    _copy_kv_buffer_to_cpu_cache[grid](
        page_num=len(page_indexes),
        mem_indexes_ptr=mem_indexes,
        page_indexes_ptr=page_indexes,
        page_readies_ptr=page_readies,
        big_page_buffer_ids=big_page_buffer_ids,
        cpu_cache_full_att=cpu_cache_full_att,
        cpu_cache_full_att_stride_p=cpu_cache_full_att.stride(0),
        cpu_cache_full_att_stride_h=cpu_cache_full_att.stride(1),
        cpu_cache_full_att_stride_d=cpu_cache_full_att.stride(2),
        cpu_cache_conv=cpu_cache_conv,
        cpu_cache_conv_stride_p=cpu_cache_conv.stride(0),
        cpu_cache_conv_stride_t=cpu_cache_conv.stride(1),
        cpu_cache_conv_stride_d=cpu_cache_conv.stride(2),
        cpu_cache_ssm=cpu_cache_ssm,
        cpu_cache_ssm_stride_p=cpu_cache_ssm.stride(0),
        cpu_cache_ssm_stride_t=cpu_cache_ssm.stride(1),
        cpu_cache_ssm_stride_d=cpu_cache_ssm.stride(2),
        gpu_kv_full_att_state=gpu_kv_full_att_state,
        gpu_kv_full_att_stride_s=gpu_kv_full_att_state.stride(0),
        gpu_kv_full_att_stride_l=gpu_kv_full_att_state.stride(1),
        gpu_kv_full_att_stride_d=gpu_kv_full_att_state.stride(2),
        cpu_kv_conv_state=cpu_kv_conv_state,
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(1),
        cpu_kv_ssm_state=cpu_kv_ssm_state,
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(1),
        gpu_full_att_tail_dim=gpu_full_att_tail_dim,
        cpu_kv_conv_tail_dim=cpu_kv_conv_tail_dim,
        cpu_kv_ssm_tail_dim=cpu_kv_ssm_tail_dim,
        tp_rank=tp_rank,
        full_att_layer_num=full_att_layer_num,
        big_page_token_num=big_page_token_num,
        head_scale_size=head_scale_size,
        BLOCK=BLOCK,
    )


@triton.jit
def _copy_cpu_cache_to_kv_buffer(
    page_num,
    mem_indexes_ptr,  # [move_token_num]
    page_indexes_ptr,  # [page_num],
    big_page_buffer_ids,  # [page_num]
    cpu_cache_full_att,  # [all_page_num, head, xdim]
    cpu_cache_full_att_stride_p,
    cpu_cache_full_att_stride_h,
    cpu_cache_full_att_stride_d,
    cpu_cache_conv,  # [all_page_num, tp_world_size, xxdim]
    cpu_cache_conv_stride_p,
    cpu_cache_conv_stride_t,
    cpu_cache_conv_stride_d,
    cpu_cache_ssm,  # [all_page_num, tp_world_size, xxxdim]
    cpu_cache_ssm_stride_p,
    cpu_cache_ssm_stride_t,
    cpu_cache_ssm_stride_d,
    gpu_kv_full_att_state,  # [token_size, full_att_layer_num, xdim]
    gpu_kv_full_att_stride_s,
    gpu_kv_full_att_stride_l,
    gpu_kv_full_att_stride_d,
    cpu_kv_conv_state,  # [buffer_count, xxxxxdim]
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_state,  # [buffer_count, xxxxxxxdim]
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_d,
    gpu_full_att_tail_dim,
    cpu_kv_conv_tail_dim,
    cpu_kv_ssm_tail_dim,
    tp_rank,
    full_att_layer_num,
    big_page_token_num,
    head_scale_size,
    BLOCK: tl.constexpr,
):
    block_index_start = tl.program_id(0)
    grid_num = tl.num_programs(0)
    # 将 所有stride 切成 tl.int64
    cpu_cache_full_att_stride_p = tl.cast(cpu_cache_full_att_stride_p, tl.int64)
    cpu_cache_full_att_stride_h = tl.cast(cpu_cache_full_att_stride_h, tl.int64)
    cpu_cache_full_att_stride_d = tl.cast(cpu_cache_full_att_stride_d, tl.int64)
    cpu_cache_conv_stride_p = tl.cast(cpu_cache_conv_stride_p, tl.int64)
    cpu_cache_conv_stride_t = tl.cast(cpu_cache_conv_stride_t, tl.int64)
    cpu_cache_conv_stride_d = tl.cast(cpu_cache_conv_stride_d, tl.int64)
    cpu_cache_ssm_stride_p = tl.cast(cpu_cache_ssm_stride_p, tl.int64)
    cpu_cache_ssm_stride_t = tl.cast(cpu_cache_ssm_stride_t, tl.int64)
    cpu_cache_ssm_stride_d = tl.cast(cpu_cache_ssm_stride_d, tl.int64)
    gpu_kv_full_att_stride_s = tl.cast(gpu_kv_full_att_stride_s, tl.int64)
    gpu_kv_full_att_stride_l = tl.cast(gpu_kv_full_att_stride_l, tl.int64)
    gpu_kv_full_att_stride_d = tl.cast(gpu_kv_full_att_stride_d, tl.int64)
    cpu_kv_conv_stride_s = tl.cast(cpu_kv_conv_stride_s, tl.int64)
    cpu_kv_conv_stride_d = tl.cast(cpu_kv_conv_stride_d, tl.int64)
    cpu_kv_ssm_stride_s = tl.cast(cpu_kv_ssm_stride_s, tl.int64)
    cpu_kv_ssm_stride_d = tl.cast(cpu_kv_ssm_stride_d, tl.int64)

    for block_index in range(block_index_start, page_num, grid_num):
        cpu_page_index = tl.load(page_indexes_ptr + block_index).to(tl.int64)

        mem_start_ptr = mem_indexes_ptr + big_page_token_num * block_index
        for i in range(tl.cdiv(gpu_full_att_tail_dim, BLOCK)):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < gpu_full_att_tail_dim
            per_token_size = gpu_full_att_tail_dim // big_page_token_num
            per_layer_size = per_token_size // full_att_layer_num
            mem_offs = gpu_start_i // (per_token_size)
            mem_index = tl.load(mem_start_ptr + mem_offs, mask=mask, other=-11111111)
            layer_index = (gpu_start_i // (per_layer_size)) % full_att_layer_num
            dim_index = gpu_start_i % per_layer_size

            src_cpu_cache_full_att_ptr = (
                cpu_cache_full_att
                + cpu_page_index * cpu_cache_full_att_stride_p
                + (tp_rank // head_scale_size) * cpu_cache_full_att_stride_h
                + gpu_start_i
            )
            cpu_full_att_data = tl.load(src_cpu_cache_full_att_ptr, mask=mask, other=0)

            tl.store(
                gpu_kv_full_att_state
                + mem_index * gpu_kv_full_att_stride_s
                + layer_index * gpu_kv_full_att_stride_l
                + dim_index * gpu_kv_full_att_stride_d,
                cpu_full_att_data,
                mask=mask,
            )

        big_page_idx = tl.load(big_page_buffer_ids + block_index)

        for i in range(tl.cdiv(cpu_kv_conv_tail_dim, BLOCK)):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < cpu_kv_conv_tail_dim

            src_cpu_cache_conv_ptr = (
                cpu_cache_conv
                + cpu_page_index * cpu_cache_conv_stride_p
                + tp_rank * cpu_cache_conv_stride_t
                + gpu_start_i
            )
            cpu_kv_conv_data = tl.load(src_cpu_cache_conv_ptr, mask=mask, other=0)

            tl.store(
                cpu_kv_conv_state + big_page_idx * cpu_kv_conv_stride_s + gpu_start_i,
                cpu_kv_conv_data,
                mask=mask,
            )

        for i in range(tl.cdiv(cpu_kv_ssm_tail_dim, BLOCK)):
            gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
            mask = gpu_start_i < cpu_kv_ssm_tail_dim

            src_cpu_cache_ssm_ptr = (
                cpu_cache_ssm + cpu_page_index * cpu_cache_ssm_stride_p + tp_rank * cpu_cache_ssm_stride_t + gpu_start_i
            )
            cpu_kv_ssm_data = tl.load(src_cpu_cache_ssm_ptr, mask=mask, other=0)

            tl.store(
                cpu_kv_ssm_state + big_page_idx * cpu_kv_ssm_stride_s + gpu_start_i,
                cpu_kv_ssm_data,
                mask=mask,
            )

    return


def copy_cpu_cache_to_kv_buffer(
    mem_indexes: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
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
    if linear_config.full_att_all_num_kv_heads % tp_world_size == 0:
        head_scale_size = 1
    else:
        head_scale_size = tp_world_size // linear_config.full_att_all_num_kv_heads

    cpu_page_num = cpu_cache_tensor.shape[0]
    cpu_cache_tensor = cpu_cache_tensor.view(cpu_page_num, -1).view(dtype=torch.uint8)
    a = linear_config.get_cpu_cache_full_att_bytes()
    b = linear_config.get_cpu_cache_conv_bytes()
    c = linear_config.get_cpu_cache_ssm_bytes()

    if head_scale_size == 1:
        cpu_cache_full_att = cpu_cache_tensor[:, 0:a].view(cpu_page_num, tp_world_size, -1)
    else:
        cpu_cache_full_att = cpu_cache_tensor[:, 0:a].view(cpu_page_num, linear_config.full_att_all_num_kv_heads, -1)

    cpu_cache_conv = cpu_cache_tensor[:, a : (a + b)].view(cpu_page_num, tp_world_size, -1)
    cpu_cache_ssm = cpu_cache_tensor[:, (a + b) : (a + b + c)].view(cpu_page_num, tp_world_size, -1)

    gpu_full_att_kv_state = gpu_full_att_kv_state.view(
        gpu_full_att_kv_state.shape[0], gpu_full_att_kv_state.shape[1], -1
    ).view(dtype=torch.uint8)
    gpu_full_att_kv_state = gpu_full_att_kv_state.permute(1, 0, 2)  # [s, layer_num, xxdim]
    cpu_kv_conv_state = cpu_kv_conv_state.view(cpu_kv_conv_state.shape[0], -1).view(dtype=torch.uint8)
    cpu_kv_ssm_state = cpu_kv_ssm_state.view(cpu_kv_ssm_state.shape[0], -1).view(dtype=torch.uint8)

    gpu_full_att_tail_dim = gpu_full_att_kv_state.shape[-1] * gpu_full_att_kv_state.shape[-2] * big_page_token_num
    cpu_kv_conv_tail_dim = cpu_kv_conv_state.shape[-1]
    cpu_kv_ssm_tail_dim = cpu_kv_ssm_state.shape[-1]
    full_att_layer_num = gpu_full_att_kv_state.shape[-2]

    assert gpu_full_att_tail_dim == cpu_cache_full_att.shape[-1]
    assert cpu_cache_conv.shape[-1] == cpu_kv_conv_state.shape[-1]
    assert cpu_cache_ssm.shape[-1] == cpu_kv_ssm_state.shape[-1]

    assert (tp_rank // head_scale_size) < linear_config.full_att_all_num_kv_heads

    grid = (grid_num,)
    _copy_cpu_cache_to_kv_buffer[grid](
        page_num=len(page_indexes),
        mem_indexes_ptr=mem_indexes,
        page_indexes_ptr=page_indexes,
        big_page_buffer_ids=big_page_buffer_ids,
        cpu_cache_full_att=cpu_cache_full_att,
        cpu_cache_full_att_stride_p=cpu_cache_full_att.stride(0),
        cpu_cache_full_att_stride_h=cpu_cache_full_att.stride(1),
        cpu_cache_full_att_stride_d=cpu_cache_full_att.stride(2),
        cpu_cache_conv=cpu_cache_conv,
        cpu_cache_conv_stride_p=cpu_cache_conv.stride(0),
        cpu_cache_conv_stride_t=cpu_cache_conv.stride(1),
        cpu_cache_conv_stride_d=cpu_cache_conv.stride(2),
        cpu_cache_ssm=cpu_cache_ssm,
        cpu_cache_ssm_stride_p=cpu_cache_ssm.stride(0),
        cpu_cache_ssm_stride_t=cpu_cache_ssm.stride(1),
        cpu_cache_ssm_stride_d=cpu_cache_ssm.stride(2),
        gpu_kv_full_att_state=gpu_full_att_kv_state,
        gpu_kv_full_att_stride_s=gpu_full_att_kv_state.stride(0),
        gpu_kv_full_att_stride_l=gpu_full_att_kv_state.stride(1),
        gpu_kv_full_att_stride_d=gpu_full_att_kv_state.stride(2),
        cpu_kv_conv_state=cpu_kv_conv_state,
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(1),
        cpu_kv_ssm_state=cpu_kv_ssm_state,
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(1),
        gpu_full_att_tail_dim=gpu_full_att_tail_dim,
        cpu_kv_conv_tail_dim=cpu_kv_conv_tail_dim,
        cpu_kv_ssm_tail_dim=cpu_kv_ssm_tail_dim,
        tp_rank=tp_rank,
        full_att_layer_num=full_att_layer_num,
        big_page_token_num=big_page_token_num,
        head_scale_size=head_scale_size,
        BLOCK=BLOCK,
    )


@triton.jit
def _copy_linear_att_state_to_linear_att_state(
    src_conv_state,
    dst_conv_state,
    src_ssm_state,
    dst_ssm_state,
    conv_size,
    ssm_size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_num = tl.num_programs(0)

    # copy conv state
    num_conv_blocks = tl.cdiv(conv_size, BLOCK)
    for i in range(pid, num_conv_blocks, grid_num):
        start = i * BLOCK + tl.arange(0, BLOCK)
        mask = start < conv_size
        data = tl.load(src_conv_state + start, mask=mask, other=0)
        tl.store(dst_conv_state + start, data, mask=mask)

    # copy ssm state
    num_ssm_blocks = tl.cdiv(ssm_size, BLOCK)
    for i in range(pid, num_ssm_blocks, grid_num):
        start = i * BLOCK + tl.arange(0, BLOCK)
        mask = start < ssm_size
        data = tl.load(src_ssm_state + start, mask=mask, other=0)
        tl.store(dst_ssm_state + start, data, mask=mask)


def copy_linear_att_state_to_linear_att_state(
    src_conv_state: torch.Tensor,
    src_ssm_state: torch.Tensor,
    dst_conv_state: torch.Tensor,
    dst_ssm_state: torch.Tensor,
    grid_num: int = 16,
):
    assert src_conv_state.shape == dst_conv_state.shape
    assert src_ssm_state.shape == dst_ssm_state.shape

    BLOCK = 4096

    src_conv_flat = src_conv_state.view(-1).view(dtype=torch.uint8)
    dst_conv_flat = dst_conv_state.view(-1).view(dtype=torch.uint8)
    src_ssm_flat = src_ssm_state.view(-1).view(dtype=torch.uint8)
    dst_ssm_flat = dst_ssm_state.view(-1).view(dtype=torch.uint8)

    conv_size = src_conv_flat.shape[0]
    ssm_size = src_ssm_flat.shape[0]

    grid = (grid_num,)
    _copy_linear_att_state_to_linear_att_state[grid](
        src_conv_state=src_conv_flat,
        dst_conv_state=dst_conv_flat,
        src_ssm_state=src_ssm_flat,
        dst_ssm_state=dst_ssm_flat,
        conv_size=conv_size,
        ssm_size=ssm_size,
        BLOCK=BLOCK,
    )

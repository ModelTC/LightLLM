import torch
import triton
import triton.language as tl


@triton.jit
def _copy_linear_att_state_to_kv_buffer(
    gpu_conv_ptr,  # [linear_layer_num, size_num, xdim]
    gpu_ssm_ptr,  # [linear_layer_num, size_num, xdim]
    cpu_kv_conv_ptr,  # [linear_layer_num, size, xdim]
    cpu_kv_ssm_ptr,  # [linear_layer_num, size, xdim]
    b_req_idx,  # [batch_size,]
    b_seq_len,  # [batch_size,]
    req_to_token_mem_index,  # [xxx, xxxx]
    req_to_stride_idx,
    req_to_stride_s,
    gpu_conv_stride_l,
    gpu_conv_stride_s,
    gpu_conv_stride_d,
    gpu_ssm_stride_l,
    gpu_ssm_stride_s,
    gpu_ssm_stride_d,
    cpu_kv_conv_stride_l,
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_stride_l,
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_d,
    mtp_step,
    big_page_token_num,
    gpu_conv_tail_dim,
    gpu_ssm_tail_dim,
    cpu_conv_tail_dim,
    cpu_ssm_tail_dim,
    BLOCK: tl.constexpr,
):
    cur_layer = tl.program_id(0).to(tl.int64)
    cur_batch_size = tl.program_id(1).to(tl.int64)
    cpu_kv_conv_stride_s = cpu_kv_conv_stride_s.to(tl.int64)
    cpu_kv_ssm_stride_s = cpu_kv_ssm_stride_s.to(tl.int64)

    cur_seq_len = tl.load(b_seq_len + cur_batch_size)
    if cur_seq_len == 0:
        return
    if cur_seq_len % big_page_token_num != 0:
        return

    cur_req_idx = tl.load(b_req_idx + cur_batch_size).to(tl.int64)
    req_to_mem_start = (
        req_to_token_mem_index + cur_req_idx * req_to_stride_idx + (cur_seq_len - big_page_token_num) * req_to_stride_s
    )
    cur_state_req_idx = (cur_req_idx * (mtp_step + 1)).to(tl.int64)

    for i in range(tl.cdiv(gpu_conv_tail_dim, BLOCK)):
        gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
        mask = gpu_start_i < gpu_conv_tail_dim
        conv_data = tl.load(
            gpu_conv_ptr
            + cur_layer * gpu_conv_stride_l
            + cur_state_req_idx * gpu_conv_stride_s
            + gpu_start_i * gpu_conv_stride_d,
            mask=mask,
        )
        dest_mem_index = tl.load(
            req_to_mem_start + (gpu_start_i // cpu_conv_tail_dim) * req_to_stride_s, mask=mask, other=-11111111
        )
        dest_conv_ptr = (
            cpu_kv_conv_ptr
            + cur_layer * cpu_kv_conv_stride_l
            + dest_mem_index * cpu_kv_conv_stride_s
            + (gpu_start_i % cpu_conv_tail_dim) * cpu_kv_conv_stride_d
        )
        tl.store(dest_conv_ptr, conv_data, mask=mask)

    for i in range(tl.cdiv(gpu_ssm_tail_dim, BLOCK)):
        gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
        mask = gpu_start_i < gpu_ssm_tail_dim
        ssm_data = tl.load(
            gpu_ssm_ptr
            + cur_layer * gpu_ssm_stride_l
            + cur_state_req_idx * gpu_ssm_stride_s
            + gpu_start_i * gpu_ssm_stride_d,
            mask=mask,
        )
        dest_mem_index = tl.load(
            req_to_mem_start + (gpu_start_i // cpu_ssm_tail_dim) * req_to_stride_s, mask=mask, other=-11111111
        )
        dest_ssm_ptr = (
            cpu_kv_ssm_ptr
            + cur_layer * cpu_kv_ssm_stride_l
            + dest_mem_index * cpu_kv_ssm_stride_s
            + (gpu_start_i % cpu_ssm_tail_dim) * cpu_kv_ssm_stride_d
        )
        tl.store(dest_ssm_ptr, ssm_data, mask=mask)

    return


def copy_linear_att_state_to_kv_buffer(
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    req_to_token_mem_index: torch.Tensor,
    gpu_conv_state: torch.Tensor,
    gpu_ssm_state: torch.Tensor,
    cpu_kv_conv_state: torch.Tensor,  # [linear_layer_num, s, dim]
    cpu_kv_ssm_state: torch.Tensor,  # [linear_layer_num, s, xdim]
    mtp_step: int,
    big_page_token_num: int,
):
    BLOCK = 4096
    gpu_conv_state = gpu_conv_state.view(gpu_conv_state.shape[0], gpu_conv_state.shape[1], -1).view(dtype=torch.uint8)
    gpu_ssm_state = gpu_ssm_state.view(gpu_ssm_state.shape[0], gpu_ssm_state.shape[1], -1).view(dtype=torch.uint8)
    gpu_conv_tail_dim = gpu_conv_state.shape[-1]
    gpu_ssm_tail_dim = gpu_ssm_state.shape[-1]
    cpu_conv_tail_dim = cpu_kv_conv_state.shape[-1]
    cpu_ssm_tail_dim = cpu_kv_ssm_state.shape[-1]
    layer_num = cpu_kv_conv_state.shape[0]
    assert gpu_conv_state.shape[0] == gpu_ssm_state.shape[0] == cpu_kv_conv_state.shape[0] == cpu_kv_ssm_state.shape[0]

    grid = (layer_num, b_seq_len.shape[0])

    _copy_linear_att_state_to_kv_buffer[grid](
        gpu_conv_ptr=gpu_conv_state,
        gpu_ssm_ptr=gpu_ssm_state,
        cpu_kv_conv_ptr=cpu_kv_conv_state,
        cpu_kv_ssm_ptr=cpu_kv_ssm_state,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        req_to_token_mem_index=req_to_token_mem_index,
        req_to_stride_idx=req_to_token_mem_index.stride(0),
        req_to_stride_s=req_to_token_mem_index.stride(1),
        gpu_conv_stride_l=gpu_conv_state.stride(0),
        gpu_conv_stride_s=gpu_conv_state.stride(1),
        gpu_conv_stride_d=gpu_conv_state.stride(2),
        gpu_ssm_stride_l=gpu_ssm_state.stride(0),
        gpu_ssm_stride_s=gpu_ssm_state.stride(1),
        gpu_ssm_stride_d=gpu_ssm_state.stride(2),
        cpu_kv_conv_stride_l=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(1),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(2),
        cpu_kv_ssm_stride_l=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(1),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(2),
        mtp_step=mtp_step,
        big_page_token_num=big_page_token_num,
        gpu_conv_tail_dim=gpu_conv_tail_dim,
        gpu_ssm_tail_dim=gpu_ssm_tail_dim,
        cpu_conv_tail_dim=cpu_conv_tail_dim,
        cpu_ssm_tail_dim=cpu_ssm_tail_dim,
        BLOCK=BLOCK,
    )


@triton.jit
def _copy_kv_buffer_to_linear_att_state(
    gpu_conv_ptr,  # [linear_layer_num, size_num, xdim]
    gpu_ssm_ptr,  # [linear_layer_num, size_num, xdim]
    cpu_kv_conv_ptr,  # [linear_layer_num, size, xdim]
    cpu_kv_ssm_ptr,  # [linear_layer_num, size, xdim]
    req_idx,  # int
    seq_len,  # int
    req_to_token_mem_index,  # [xxx, xxxx]
    req_to_stride_idx,
    req_to_stride_s,
    gpu_conv_stride_l,
    gpu_conv_stride_s,
    gpu_conv_stride_d,
    gpu_ssm_stride_l,
    gpu_ssm_stride_s,
    gpu_ssm_stride_d,
    cpu_kv_conv_stride_l,
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_stride_l,
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_d,
    mtp_step,
    big_page_token_num,
    gpu_conv_tail_dim,
    gpu_ssm_tail_dim,
    cpu_conv_tail_dim,
    cpu_ssm_tail_dim,
    BLOCK: tl.constexpr,
):
    cur_layer = tl.program_id(0).to(tl.int64)
    cpu_kv_conv_stride_s = cpu_kv_conv_stride_s.to(tl.int64)
    cpu_kv_ssm_stride_s = cpu_kv_ssm_stride_s.to(tl.int64)
    req_idx = tl.cast(req_idx, dtype=tl.int64)

    cur_seq_len = seq_len
    if cur_seq_len == 0:
        return
    if (cur_seq_len % big_page_token_num) != 0:
        return

    cur_req_idx = req_idx
    cur_state_req_idx = (cur_req_idx * (mtp_step + 1)).to(tl.int64)

    req_to_mem_start = (
        req_to_token_mem_index + cur_req_idx * req_to_stride_idx + (cur_seq_len - big_page_token_num) * req_to_stride_s
    )

    for i in range(tl.cdiv(gpu_conv_tail_dim, BLOCK)):
        gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
        mask = gpu_start_i < gpu_conv_tail_dim
        src_mem_index = tl.load(
            req_to_mem_start + (gpu_start_i // cpu_conv_tail_dim) * req_to_stride_s, mask=mask, other=-11111111
        )
        src_conv_ptr = (
            cpu_kv_conv_ptr
            + cur_layer * cpu_kv_conv_stride_l
            + src_mem_index * cpu_kv_conv_stride_s
            + (gpu_start_i % cpu_conv_tail_dim) * cpu_kv_conv_stride_d
        )
        conv_data = tl.load(src_conv_ptr, mask=mask, other=0)
        tl.store(
            gpu_conv_ptr
            + cur_layer * gpu_conv_stride_l
            + cur_state_req_idx * gpu_conv_stride_s
            + gpu_start_i * gpu_conv_stride_d,
            conv_data,
            mask=mask,
        )

    for i in range(tl.cdiv(gpu_ssm_tail_dim, BLOCK)):
        gpu_start_i = i * BLOCK + tl.arange(0, BLOCK)
        mask = gpu_start_i < gpu_ssm_tail_dim
        src_mem_index = tl.load(
            req_to_mem_start + (gpu_start_i // cpu_ssm_tail_dim) * req_to_stride_s, mask=mask, other=-11111111
        )
        src_conv_ptr = (
            cpu_kv_ssm_ptr
            + cur_layer * cpu_kv_ssm_stride_l
            + src_mem_index * cpu_kv_ssm_stride_s
            + (gpu_start_i % cpu_ssm_tail_dim) * cpu_kv_ssm_stride_d
        )
        ssm_data = tl.load(src_conv_ptr, mask=mask, other=0)
        tl.store(
            gpu_ssm_ptr
            + cur_layer * gpu_ssm_stride_l
            + cur_state_req_idx * gpu_ssm_stride_s
            + gpu_start_i * gpu_ssm_stride_d,
            ssm_data,
            mask=mask,
        )
    return


def copy_kv_buffer_to_linear_att_state(
    req_idx: int,
    seq_len: int,
    req_to_token_mem_index: torch.Tensor,
    gpu_conv_state: torch.Tensor,
    gpu_ssm_state: torch.Tensor,
    cpu_kv_conv_state: torch.Tensor,  # [linear_layer_num, s, dim]
    cpu_kv_ssm_state: torch.Tensor,  # [linear_layer_num, s, xdim]
    mtp_step: int,
    big_page_token_num: int,
):
    BLOCK = 4096
    gpu_conv_state = gpu_conv_state.view(gpu_conv_state.shape[0], gpu_conv_state.shape[1], -1).view(dtype=torch.uint8)
    gpu_ssm_state = gpu_ssm_state.view(gpu_ssm_state.shape[0], gpu_ssm_state.shape[1], -1).view(dtype=torch.uint8)
    gpu_conv_tail_dim = gpu_conv_state.shape[-1]
    gpu_ssm_tail_dim = gpu_ssm_state.shape[-1]
    cpu_conv_tail_dim = cpu_kv_conv_state.shape[-1]
    cpu_ssm_tail_dim = cpu_kv_ssm_state.shape[-1]
    layer_num = cpu_kv_conv_state.shape[0]
    assert gpu_conv_state.shape[0] == gpu_ssm_state.shape[0] == cpu_kv_conv_state.shape[0] == cpu_kv_ssm_state.shape[0]

    grid = (layer_num,)

    _copy_kv_buffer_to_linear_att_state[grid](
        gpu_conv_ptr=gpu_conv_state,
        gpu_ssm_ptr=gpu_ssm_state,
        cpu_kv_conv_ptr=cpu_kv_conv_state,
        cpu_kv_ssm_ptr=cpu_kv_ssm_state,
        req_idx=req_idx,
        seq_len=seq_len,
        req_to_token_mem_index=req_to_token_mem_index,
        req_to_stride_idx=req_to_token_mem_index.stride(0),
        req_to_stride_s=req_to_token_mem_index.stride(1),
        gpu_conv_stride_l=gpu_conv_state.stride(0),
        gpu_conv_stride_s=gpu_conv_state.stride(1),
        gpu_conv_stride_d=gpu_conv_state.stride(2),
        gpu_ssm_stride_l=gpu_ssm_state.stride(0),
        gpu_ssm_stride_s=gpu_ssm_state.stride(1),
        gpu_ssm_stride_d=gpu_ssm_state.stride(2),
        cpu_kv_conv_stride_l=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(1),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(2),
        cpu_kv_ssm_stride_l=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(1),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(2),
        mtp_step=mtp_step,
        big_page_token_num=big_page_token_num,
        gpu_conv_tail_dim=gpu_conv_tail_dim,
        gpu_ssm_tail_dim=gpu_ssm_tail_dim,
        cpu_conv_tail_dim=cpu_conv_tail_dim,
        cpu_ssm_tail_dim=cpu_ssm_tail_dim,
        BLOCK=BLOCK,
    )

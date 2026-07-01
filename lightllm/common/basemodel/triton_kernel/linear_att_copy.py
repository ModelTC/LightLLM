import torch
import triton
import triton.language as tl


@triton.jit
def _copy_linear_att_state_to_kv_buffer(
    gpu_conv_ptr,  # uint8 view: [linear_layer_num, req_num, conv_dim, gpu_conv_row_bytes]
    gpu_ssm_ptr,  # uint8 view: [linear_layer_num, req_num * (mtp_step + 1), ssm_bytes]
    cpu_kv_conv_ptr,  # uint8 view: [buffer_num, linear_layer_num, conv_dim, cpu_conv_row_bytes]
    cpu_kv_ssm_ptr,  # uint8 view: [buffer_num, linear_layer_num, ssm_bytes]
    b_req_idx,  # [batch_size,]
    big_page_buffer_ids,  # [batch_size,]
    gpu_conv_stride_l,
    gpu_conv_stride_s,
    gpu_conv_stride_d,
    gpu_ssm_stride_l,
    gpu_ssm_stride_s,
    gpu_ssm_stride_d,
    cpu_kv_conv_stride_s,
    cpu_kv_conv_stride_l,
    cpu_kv_conv_stride_d,
    cpu_kv_ssm_stride_s,
    cpu_kv_ssm_stride_l,
    cpu_kv_ssm_stride_d,
    mtp_step,
    conv_dim,  # number of conv rows
    cpu_conv_row_bytes,  # bytes copied per conv row; equals the CPU/cache row width
    gpu_ssm_tail_dim,
    BLOCK: tl.constexpr,
):
    cur_layer = tl.program_id(0).to(tl.int64)
    cur_batch = tl.program_id(1).to(tl.int64)
    cpu_kv_conv_stride_s = tl.cast(cpu_kv_conv_stride_s, dtype=tl.int64)
    cpu_kv_ssm_stride_s = tl.cast(cpu_kv_ssm_stride_s, dtype=tl.int64)
    gpu_conv_stride_s = tl.cast(gpu_conv_stride_s, dtype=tl.int64)
    gpu_conv_stride_d = tl.cast(gpu_conv_stride_d, dtype=tl.int64)
    gpu_ssm_stride_s = tl.cast(gpu_ssm_stride_s, dtype=tl.int64)
    cpu_kv_conv_stride_d = tl.cast(cpu_kv_conv_stride_d, dtype=tl.int64)
    cpu_conv_row_bytes = tl.cast(cpu_conv_row_bytes, dtype=tl.int64)

    big_page_buffer_idx = tl.load(big_page_buffer_ids + cur_batch)
    if big_page_buffer_idx == -1:
        return

    cur_req_idx = tl.load(b_req_idx + cur_batch).to(tl.int64)
    cur_state_req_idx = (cur_req_idx * (mtp_step + 1)).to(tl.int64)

    gpu_conv_base = gpu_conv_ptr + cur_layer * gpu_conv_stride_l + cur_req_idx * gpu_conv_stride_s
    cpu_conv_base = cpu_kv_conv_ptr + big_page_buffer_idx * cpu_kv_conv_stride_s + cur_layer * cpu_kv_conv_stride_l
    conv_tail_dim = conv_dim * cpu_conv_row_bytes
    for i in range(tl.cdiv(conv_tail_dim, BLOCK)):
        conv_start = i * BLOCK + tl.arange(0, BLOCK)
        conv_row = conv_start // cpu_conv_row_bytes
        conv_col = conv_start - conv_row * cpu_conv_row_bytes
        mask = conv_row < conv_dim
        conv_data = tl.load(gpu_conv_base + conv_row * gpu_conv_stride_d + conv_col, mask=mask)
        tl.store(cpu_conv_base + conv_row * cpu_kv_conv_stride_d + conv_col, conv_data, mask=mask)

    for i in range(tl.cdiv(gpu_ssm_tail_dim, BLOCK)):
        gpu_start_off = i * BLOCK + tl.arange(0, BLOCK)
        mask = gpu_start_off < gpu_ssm_tail_dim
        ssm_data = tl.load(
            gpu_ssm_ptr + cur_layer * gpu_ssm_stride_l + cur_state_req_idx * gpu_ssm_stride_s + gpu_start_off,
            mask=mask,
        )
        dest_ssm_ptr = (
            cpu_kv_ssm_ptr + big_page_buffer_idx * cpu_kv_ssm_stride_s + cur_layer * cpu_kv_ssm_stride_l + gpu_start_off
        )
        tl.store(dest_ssm_ptr, ssm_data, mask=mask)

    return


def copy_linear_att_state_to_kv_buffer(
    b_req_idx: torch.Tensor,
    big_page_buffer_ids: torch.Tensor,
    gpu_conv_state: torch.Tensor,  # [linear_layer_num, req_num, conv_dim, widened_width]
    gpu_ssm_state: torch.Tensor,  # [linear_layer_num, req_num * (mtp_step + 1), ...]
    cpu_kv_conv_state: torch.Tensor,  # [buffer_num, linear_layer_num, conv_dim, base_width]
    cpu_kv_ssm_state: torch.Tensor,  # [buffer_num, linear_layer_num, ...]
    mtp_step: int,
):
    assert len(b_req_idx) == big_page_buffer_ids.shape[0]
    BLOCK = 4096

    assert gpu_conv_state.dim() >= 4, "gpu_conv_state must be [layer, s, conv_dim, widened_width]"
    assert cpu_kv_conv_state.dim() >= 4, "cpu_kv_conv_state must be [size, layer, conv_dim, width_narrow]"
    gpu_conv_state = gpu_conv_state.view(
        gpu_conv_state.shape[0], gpu_conv_state.shape[1], gpu_conv_state.shape[2], -1
    ).view(dtype=torch.uint8)
    cpu_kv_conv_state = cpu_kv_conv_state.view(
        cpu_kv_conv_state.shape[0], cpu_kv_conv_state.shape[1], cpu_kv_conv_state.shape[2], -1
    ).view(dtype=torch.uint8)
    gpu_ssm_state = gpu_ssm_state.view(gpu_ssm_state.shape[0], gpu_ssm_state.shape[1], -1).view(dtype=torch.uint8)
    cpu_kv_ssm_state = cpu_kv_ssm_state.view(cpu_kv_ssm_state.shape[0], cpu_kv_ssm_state.shape[1], -1).view(
        dtype=torch.uint8
    )
    assert gpu_conv_state.shape[2] == cpu_kv_conv_state.shape[2], "conv_dim mismatch between gpu and cpu conv buffers"
    assert gpu_ssm_state.shape[-1] == cpu_kv_ssm_state.shape[-1]

    conv_dim = gpu_conv_state.shape[2]
    cpu_conv_row_bytes = cpu_kv_conv_state.shape[-1]
    assert cpu_conv_row_bytes <= gpu_conv_state.shape[-1]
    gpu_ssm_tail_dim = gpu_ssm_state.shape[-1]

    layer_num = gpu_conv_state.shape[0]

    grid = (layer_num, b_req_idx.shape[0])

    _copy_linear_att_state_to_kv_buffer[grid](
        gpu_conv_ptr=gpu_conv_state,
        gpu_ssm_ptr=gpu_ssm_state,
        cpu_kv_conv_ptr=cpu_kv_conv_state,
        cpu_kv_ssm_ptr=cpu_kv_ssm_state,
        b_req_idx=b_req_idx,
        big_page_buffer_ids=big_page_buffer_ids,
        gpu_conv_stride_l=gpu_conv_state.stride(0),
        gpu_conv_stride_s=gpu_conv_state.stride(1),
        gpu_conv_stride_d=gpu_conv_state.stride(2),
        gpu_ssm_stride_l=gpu_ssm_state.stride(0),
        gpu_ssm_stride_s=gpu_ssm_state.stride(1),
        gpu_ssm_stride_d=gpu_ssm_state.stride(2),
        cpu_kv_conv_stride_s=cpu_kv_conv_state.stride(0),
        cpu_kv_conv_stride_l=cpu_kv_conv_state.stride(1),
        cpu_kv_conv_stride_d=cpu_kv_conv_state.stride(2),
        cpu_kv_ssm_stride_s=cpu_kv_ssm_state.stride(0),
        cpu_kv_ssm_stride_l=cpu_kv_ssm_state.stride(1),
        cpu_kv_ssm_stride_d=cpu_kv_ssm_state.stride(2),
        mtp_step=mtp_step,
        conv_dim=conv_dim,
        cpu_conv_row_bytes=cpu_conv_row_bytes,
        gpu_ssm_tail_dim=gpu_ssm_tail_dim,
        BLOCK=BLOCK,
    )

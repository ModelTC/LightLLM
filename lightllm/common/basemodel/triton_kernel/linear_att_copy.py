import torch
import triton
import triton.language as tl


@triton.jit
def _copy_linear_att_state_to_kv_buffer(
    gpu_conv_ptr,  # [linear_layer_num, size_num, conv_dim * gpu_widened_width] (uint8 tail)
    gpu_ssm_ptr,  # [linear_layer_num, size_num, xxdim]
    cpu_kv_conv_ptr,  # [size, linear_layer_num, conv_dim * width_narrow] (uint8 tail)
    cpu_kv_ssm_ptr,  # [size, linear_layer_num, xxdim]
    b_req_idx,  # [batch_size,]
    big_page_buffer_ids,  # [batch_size,]
    num_accepted_tokens_ptr,  # [batch_size,]
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
    conv_dim,  # number of conv rows (the d dimension)
    gpu_conv_row_bytes,  # widened per-row byte length: gpu_widened_width * itemsize
    conv_narrow_row_bytes,  # narrow per-row byte length: width_narrow * itemsize
    gpu_ssm_tail_dim,
    BLOCK: tl.constexpr,
):
    cur_layer = tl.program_id(0).to(tl.int64)
    cur_batch = tl.program_id(1).to(tl.int64)
    cpu_kv_conv_stride_s = tl.cast(cpu_kv_conv_stride_s, dtype=tl.int64)
    cpu_kv_ssm_stride_s = tl.cast(cpu_kv_ssm_stride_s, dtype=tl.int64)
    gpu_conv_stride_s = tl.cast(gpu_conv_stride_s, dtype=tl.int64)
    gpu_ssm_stride_s = tl.cast(gpu_ssm_stride_s, dtype=tl.int64)

    big_page_buffer_idx = tl.load(big_page_buffer_ids + cur_batch)
    if big_page_buffer_idx == -1:
        return

    cur_req_idx = tl.load(b_req_idx + cur_batch).to(tl.int64)
    accept_len = tl.load(num_accepted_tokens_ptr + cur_batch).to(tl.int64)
    canonical_off = accept_len - 1

    conv_src_slot = cur_req_idx
    conv_off_bytes = canonical_off * gpu_conv_stride_d
    gpu_conv_base = gpu_conv_ptr + cur_layer * gpu_conv_stride_l + conv_src_slot * gpu_conv_stride_s + conv_off_bytes
    cpu_conv_base = cpu_kv_conv_ptr + big_page_buffer_idx * cpu_kv_conv_stride_s + cur_layer * cpu_kv_conv_stride_l
    for d in range(conv_dim):
        for i in range(tl.cdiv(conv_narrow_row_bytes, BLOCK)):
            off = i * BLOCK + tl.arange(0, BLOCK)
            mask = off < conv_narrow_row_bytes
            conv_data = tl.load(gpu_conv_base + d * gpu_conv_row_bytes + off, mask=mask)
            tl.store(cpu_conv_base + d * cpu_kv_conv_stride_d + off, conv_data, mask=mask)

    ssm_src_slot = (cur_req_idx * (mtp_step + 1) + canonical_off).to(tl.int64)
    for i in range(tl.cdiv(gpu_ssm_tail_dim, BLOCK)):
        gpu_start_off = i * BLOCK + tl.arange(0, BLOCK)
        mask = gpu_start_off < gpu_ssm_tail_dim
        ssm_data = tl.load(
            gpu_ssm_ptr + cur_layer * gpu_ssm_stride_l + ssm_src_slot * gpu_ssm_stride_s + gpu_start_off,
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
    gpu_conv_state: torch.Tensor,  # [linear_layer_num, s_widened, conv_dim, gpu_widened_width]
    gpu_ssm_state: torch.Tensor,  # [linear_layer_num, s_block, ...]
    cpu_kv_conv_state: torch.Tensor,  # [size, linear_layer_num, conv_dim, width_narrow]
    cpu_kv_ssm_state: torch.Tensor,  # [size, linear_layer_num, ...]
    mtp_step: int,
    b_num_accepted_tokens: torch.Tensor,  # [batch_size,] per-req post-accept count (>=1)
):
    assert len(b_req_idx) == big_page_buffer_ids.shape[0]
    assert len(b_req_idx) == b_num_accepted_tokens.shape[0]
    BLOCK = 4096

    assert gpu_conv_state.dim() >= 4, "gpu_conv_state must be [layer, s, conv_dim, widened_width]"
    assert cpu_kv_conv_state.dim() >= 4, "cpu_kv_conv_state must be [size, layer, conv_dim, width_narrow]"
    # #6: the byte snapshot hardcodes gpu_conv_stride_d=conv_itemsize, which is only valid when the
    # widened-width axis is element-contiguous (stride 1). Fail fast instead of snapshotting wrong bytes.
    assert gpu_conv_state.stride(3) == 1, (
        "gpu_conv_state widened-width axis must be element-contiguous (stride 1); "
        "gpu_conv_stride_d=conv_itemsize assumes it"
    )
    # #18: canonical_off = accept_len - 1 indexes into the widened slot; bound it to [0, mtp_step]
    # (accept_len in [1, mtp_step+1]) so a stale/oversized accept-count can't slice past the slot.
    assert int(b_num_accepted_tokens.min()) >= 1 and int(b_num_accepted_tokens.max()) <= mtp_step + 1, (
        f"b_num_accepted_tokens out of range [1, {mtp_step + 1}]: "
        f"min={int(b_num_accepted_tokens.min())} max={int(b_num_accepted_tokens.max())}"
    )
    conv_itemsize = gpu_conv_state.element_size()
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
    gpu_conv_row_bytes = gpu_conv_state.shape[-1]  # widened per-row byte length
    conv_narrow_row_bytes = cpu_kv_conv_state.shape[-1]  # narrow per-row byte length
    assert conv_narrow_row_bytes <= gpu_conv_row_bytes
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
        num_accepted_tokens_ptr=b_num_accepted_tokens,
        gpu_conv_stride_l=gpu_conv_state.stride(0),
        gpu_conv_stride_s=gpu_conv_state.stride(1),
        gpu_conv_stride_d=conv_itemsize,
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
        gpu_conv_row_bytes=gpu_conv_row_bytes,
        conv_narrow_row_bytes=conv_narrow_row_bytes,
        gpu_ssm_tail_dim=gpu_ssm_tail_dim,
        BLOCK=BLOCK,
    )

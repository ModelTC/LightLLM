import torch
import triton
import triton.language as tl


@triton.jit
def _build_prefill_row_table_kernel(
    prefill_mem_index_ptr,
    row_table_ptr,
    prefill_token_num,
):
    pid = tl.program_id(0)
    if pid < prefill_token_num:
        mem_index = tl.load(prefill_mem_index_ptr + pid)
        tl.store(row_table_ptr + mem_index, pid)


@triton.jit
def _fill_compact_kv_kernel(
    packed_nope_ptr,
    packed_scale_ptr,
    packed_rope_ptr,
    unique_mem_index_ptr,
    prefill_row_table_ptr,
    prefill_kv_ptr,
    compact_kv_ptr,
    packed_nope_stride_s,
    packed_nope_stride_d,
    packed_scale_stride_s,
    packed_scale_stride_d,
    packed_rope_stride_s,
    packed_rope_stride_d,
    prefill_kv_stride_s,
    prefill_kv_stride_d,
    compact_kv_stride_s,
    compact_kv_stride_d,
    unique_num,
    KV_NOPE_DIM: tl.constexpr,
    KV_ROPE_DIM: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_block = tl.program_id(1)

    if pid_s >= unique_num:
        return

    mem_index = tl.load(unique_mem_index_ptr + pid_s)
    prefill_row = tl.load(prefill_row_table_ptr + mem_index)
    offs_d = tl.arange(0, BLOCK_D)

    if prefill_row != -1:
        if pid_block < (KV_NOPE_DIM // GROUP_SIZE):
            mask = offs_d < GROUP_SIZE
            value = tl.load(
                prefill_kv_ptr
                + prefill_row * prefill_kv_stride_s
                + (pid_block * GROUP_SIZE + offs_d) * prefill_kv_stride_d,
                mask=mask,
            ).to(tl.float32)
            tl.store(
                compact_kv_ptr + pid_s * compact_kv_stride_s + (pid_block * GROUP_SIZE + offs_d) * compact_kv_stride_d,
                value,
                mask=mask,
            )
        else:
            mask = offs_d < KV_ROPE_DIM
            value = tl.load(
                prefill_kv_ptr + prefill_row * prefill_kv_stride_s + (KV_NOPE_DIM + offs_d) * prefill_kv_stride_d,
                mask=mask,
            ).to(tl.float32)
            tl.store(
                compact_kv_ptr + pid_s * compact_kv_stride_s + (KV_NOPE_DIM + offs_d) * compact_kv_stride_d,
                value,
                mask=mask,
            )
    else:
        if pid_block < (KV_NOPE_DIM // GROUP_SIZE):
            mask = offs_d < GROUP_SIZE
            src_fp8 = tl.load(
                packed_nope_ptr
                + mem_index * packed_nope_stride_s
                + (pid_block * GROUP_SIZE + offs_d) * packed_nope_stride_d,
                mask=mask,
            )
            scale = tl.load(packed_scale_ptr + mem_index * packed_scale_stride_s + pid_block * packed_scale_stride_d)
            value = src_fp8.to(tl.float32) * scale
            tl.store(
                compact_kv_ptr + pid_s * compact_kv_stride_s + (pid_block * GROUP_SIZE + offs_d) * compact_kv_stride_d,
                value,
                mask=mask,
            )
        else:
            mask = offs_d < KV_ROPE_DIM
            value = tl.load(
                packed_rope_ptr + mem_index * packed_rope_stride_s + offs_d * packed_rope_stride_d,
                mask=mask,
            ).to(tl.float32)
            tl.store(
                compact_kv_ptr + pid_s * compact_kv_stride_s + (KV_NOPE_DIM + offs_d) * compact_kv_stride_d,
                value,
                mask=mask,
            )


@torch.no_grad()
def get_prefill_kv_cache_and_remap_indices_triton(
    packed_kv: torch.Tensor,
    topk_mem_indices: torch.Tensor,
    prefill_mem_index: torch.Tensor,
    prefill_cache_kv: torch.Tensor,
    prefill_dtype: torch.dtype,
):
    squeeze_h_kv = topk_mem_indices.ndim == 2
    if squeeze_h_kv:
        topk_mem_indices = topk_mem_indices.unsqueeze(1)

    original_shape = topk_mem_indices.shape
    flat_topk = topk_mem_indices.reshape(-1).contiguous().to(torch.int32)
    valid_mask = flat_topk != -1
    valid_topk = flat_topk[valid_mask]
    table_size = packed_kv.shape[0]

    prefill_row_table = torch.full((table_size,), -1, dtype=torch.int32, device=packed_kv.device)
    _build_prefill_row_table_kernel[(prefill_mem_index.numel(),)](
        prefill_mem_index_ptr=prefill_mem_index.to(torch.int32).contiguous(),
        row_table_ptr=prefill_row_table,
        prefill_token_num=prefill_mem_index.numel(),
        num_warps=4,
    )

    unique_mem_index, inverse = torch.unique(valid_topk, sorted=False, return_inverse=True)
    unique_mem_index = unique_mem_index.to(torch.int32)
    unique_count = unique_mem_index.numel()
    remapped_flat = torch.full_like(flat_topk, -1)
    remapped_flat[valid_mask] = inverse.to(torch.int32)

    compact_kv = torch.empty((unique_count, 1, 576), dtype=prefill_dtype, device=packed_kv.device)
    packed_nope = packed_kv[:, :, :512].view(torch.float8_e4m3fn).view(-1, 512)
    packed_scale = packed_kv[:, :, 512:528].view(torch.float32).view(-1, 4)
    packed_rope = packed_kv[:, :, 528:].view(torch.bfloat16).view(-1, 64)
    prefill_kv_2d = prefill_cache_kv.view(-1, 576)
    compact_kv_2d = compact_kv.view(-1, 576)

    _fill_compact_kv_kernel[(unique_count, 5)](
        packed_nope_ptr=packed_nope,
        packed_scale_ptr=packed_scale,
        packed_rope_ptr=packed_rope,
        unique_mem_index_ptr=unique_mem_index,
        prefill_row_table_ptr=prefill_row_table,
        prefill_kv_ptr=prefill_kv_2d,
        compact_kv_ptr=compact_kv_2d,
        packed_nope_stride_s=packed_nope.stride(0),
        packed_nope_stride_d=packed_nope.stride(1),
        packed_scale_stride_s=packed_scale.stride(0),
        packed_scale_stride_d=packed_scale.stride(1),
        packed_rope_stride_s=packed_rope.stride(0),
        packed_rope_stride_d=packed_rope.stride(1),
        prefill_kv_stride_s=prefill_kv_2d.stride(0),
        prefill_kv_stride_d=prefill_kv_2d.stride(1),
        compact_kv_stride_s=compact_kv_2d.stride(0),
        compact_kv_stride_d=compact_kv_2d.stride(1),
        unique_num=unique_count,
        KV_NOPE_DIM=512,
        KV_ROPE_DIM=64,
        GROUP_SIZE=128,
        BLOCK_D=128,
        num_warps=4,
    )

    remapped = remapped_flat.view(original_shape)
    if squeeze_h_kv:
        remapped = remapped.squeeze(1)
    return compact_kv, remapped

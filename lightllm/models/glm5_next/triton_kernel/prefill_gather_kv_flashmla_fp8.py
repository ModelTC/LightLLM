import torch
import triton
import triton.language as tl


@triton.jit
def _build_prefill_row_table_kernel(PrefillMemIndex, RowTable):
    row = tl.program_id(0)
    mem_index = tl.load(PrefillMemIndex + row)
    tl.store(RowTable + mem_index, row)


@triton.jit
def _gather_prefill_kv_kernel(
    PackedKV,
    MemIndex,
    PrefillRowTable,
    PrefillKV,
    Output,
    packed_stride_s,
    prefill_stride_s,
    prefill_stride_d,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = group * 128 + tl.arange(0, 128)
    mem_index = tl.load(MemIndex + row).to(tl.int64)
    prefill_row = tl.load(PrefillRowTable + mem_index)

    if prefill_row != -1:
        values = tl.load(PrefillKV + prefill_row * prefill_stride_s + offsets * prefill_stride_d).to(tl.float32)
    else:
        packed_row = PackedKV + mem_index * packed_stride_s
        quantized = tl.load(packed_row.to(tl.pointer_type(tl.float8e4nv)) + offsets)
        scale = tl.load((packed_row + 512).to(tl.pointer_type(tl.float32)) + group)
        values = quantized.to(tl.float32) * scale
    tl.store(Output + row * 512 + offsets, values)


@torch.no_grad()
def gather_prefill_kv_cache_triton(
    packed_kv: torch.Tensor,
    mem_index: torch.Tensor,
    prefill_mem_index: torch.Tensor,
    prefill_cache_kv: torch.Tensor,
):
    """Gather NoPE KV in ragged order, preserving the precision of the current prefill block."""
    assert prefill_cache_kv.shape[1:] == (1, 512)
    assert packed_kv.dtype == torch.uint8 and packed_kv.shape[1] == 1
    assert packed_kv.shape[2] >= 656 and packed_kv.stride(2) == 1
    prefill_row_table = torch.full((packed_kv.shape[0],), -1, dtype=torch.int32, device=packed_kv.device)
    _build_prefill_row_table_kernel[(prefill_mem_index.numel(),)](
        prefill_mem_index.contiguous(), prefill_row_table, num_warps=4
    )
    output = torch.empty((mem_index.numel(), 1, 512), dtype=prefill_cache_kv.dtype, device=packed_kv.device)
    _gather_prefill_kv_kernel[(mem_index.numel(), 4)](
        packed_kv,
        mem_index.contiguous(),
        prefill_row_table,
        prefill_cache_kv,
        output,
        packed_kv.stride(0),
        prefill_cache_kv.stride(0),
        prefill_cache_kv.stride(2),
        num_warps=4,
    )
    return output

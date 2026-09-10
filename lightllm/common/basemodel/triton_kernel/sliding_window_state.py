import torch
import triton
import triton.language as tl


@triton.jit
def _copy_sliding_window_checkpoint(
    gpu_sliding_kv_ptr,  # uint8 view: [layer, token_capacity, ...]
    req_to_sliding_window,  # [req_num, max_seq_len], absolute token position -> GPU mem_index
    cache_len,
    cpu_kv_sliding_ptr,  # uint8 view of one checkpoint: [layer, window, ...]
    req_idx,
    req_table_stride,
    gpu_sliding_layer_stride_bytes,
    gpu_sliding_token_stride_bytes,
    SLIDING_WINDOW: tl.constexpr,
    KV_TOKEN_BYTES: tl.constexpr,
    CPU_STATE_BYTES: tl.constexpr,
    RESTORE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    for block_index in range(tl.program_id(0), tl.cdiv(CPU_STATE_BYTES, BLOCK_BYTES), tl.num_programs(0)):
        state_byte_offsets = tl.cast(block_index, tl.int64) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        layer_index = state_byte_offsets // (SLIDING_WINDOW * KV_TOKEN_BYTES)
        window_offset = state_byte_offsets // KV_TOKEN_BYTES % SLIDING_WINDOW
        # CPU checkpoints retain absolute token position % W ordering, independent of GPU slot allocation.
        token_position = cache_len - 1 - (cache_len - 1 - window_offset + SLIDING_WINDOW) % SLIDING_WINDOW
        valid_token = (state_byte_offsets < CPU_STATE_BYTES) & (token_position >= 0)
        mem_index = tl.load(
            req_to_sliding_window + tl.cast(req_idx, tl.int64) * req_table_stride + token_position, valid_token, other=0
        ).to(tl.int64)
        gpu_kv_ptr = (
            gpu_sliding_kv_ptr
            + layer_index * gpu_sliding_layer_stride_bytes
            + mem_index * gpu_sliding_token_stride_bytes
            + state_byte_offsets % KV_TOKEN_BYTES
        )
        if RESTORE:
            kv_data = tl.load(cpu_kv_sliding_ptr + state_byte_offsets, valid_token, other=0)
            tl.store(gpu_kv_ptr, kv_data, valid_token)
        else:
            kv_data = tl.load(gpu_kv_ptr, valid_token, other=0)
            tl.store(cpu_kv_sliding_ptr + state_byte_offsets, kv_data, state_byte_offsets < CPU_STATE_BYTES)


def copy_sliding_window_checkpoint(
    gpu_sliding_kv_buffer: torch.Tensor,
    req_to_sliding_window: torch.Tensor,
    cache_len: int,
    req_idx: int,
    cpu_kv_sliding_state: torch.Tensor,
    restore: bool = False,
):
    """GPU [layer, token_capacity, ...] 与 CPU pinned [layer, window, ...] checkpoint 的逐字节拷贝。

    restore=True: CPU checkpoint → GPU KV；否则 GPU KV → CPU checkpoint。
    cache_len 是窗口的绝对 token 右边界；KV_TOKEN_BYTES 是每层每个 token 的 KV 字节数。
    """
    gpu_sliding_kv_bytes = gpu_sliding_kv_buffer.view(torch.uint8)
    cpu_kv_sliding_bytes = cpu_kv_sliding_state.view(torch.uint8)
    _copy_sliding_window_checkpoint[(16,)](
        gpu_sliding_kv_ptr=gpu_sliding_kv_bytes,
        req_to_sliding_window=req_to_sliding_window,
        cache_len=cache_len,
        cpu_kv_sliding_ptr=cpu_kv_sliding_bytes,
        req_idx=req_idx,
        req_table_stride=req_to_sliding_window.stride(0),
        gpu_sliding_layer_stride_bytes=gpu_sliding_kv_bytes.stride(0),
        gpu_sliding_token_stride_bytes=gpu_sliding_kv_bytes.stride(1),
        SLIDING_WINDOW=cpu_kv_sliding_state.shape[1],
        KV_TOKEN_BYTES=gpu_sliding_kv_bytes.stride(1),
        CPU_STATE_BYTES=cpu_kv_sliding_bytes.numel(),
        RESTORE=restore,
        BLOCK_BYTES=4096,
    )

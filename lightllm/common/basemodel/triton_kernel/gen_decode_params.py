import torch
import triton
import triton.language as tl


@triton.jit
def _gen_decode_params_kernel(seq_len, q_len, cu_q_len, cu_kv_len, position, rows, seq_stride, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(cu_q_len, 0)
    tl.store(cu_kv_len, 0)
    carry = tl.full((), 0, tl.int64)
    for start in range(0, rows, BLOCK):
        row = start + lane
        valid = row < rows
        length = tl.load(seq_len + row * seq_stride, valid, 0)
        scanned = tl.cumsum(length.to(tl.int32)) + carry
        tl.store(q_len + row, 1, valid)
        tl.store(cu_q_len + row + 1, row + 1, valid)
        tl.store(cu_kv_len + row + 1, scanned, valid)
        tl.store(position + row, length - 1, valid)
        carry = tl.sum(length.to(tl.int64), 0) + carry


@torch.no_grad()
def gen_decode_params(b_seq_len: torch.Tensor):
    rows = b_seq_len.shape[0]
    b_q_seq_len = torch.empty((rows,), dtype=torch.int32, device=b_seq_len.device)
    b1_cu_q_seq_len = torch.empty((rows + 1,), dtype=torch.int32, device=b_seq_len.device)
    b1_cu_kv_seq_len = torch.empty_like(b1_cu_q_seq_len)
    position_ids = torch.empty_like(b_seq_len)
    _gen_decode_params_kernel[(1,)](
        b_seq_len,
        b_q_seq_len,
        b1_cu_q_seq_len,
        b1_cu_kv_seq_len,
        position_ids,
        rows,
        b_seq_len.stride(0),
        min(1024, triton.next_power_of_2(max(32, rows))),
        num_warps=4,
    )
    return b_q_seq_len, b1_cu_q_seq_len, b_seq_len, b1_cu_kv_seq_len, position_ids

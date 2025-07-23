import torch

import triton
import triton.language as tl


@triton.jit
def _per_head_max_reduce_kernel(
    Q,
    Scales,
    StartLoc,
    stride_q_t,
    stride_q_h,
    stride_scales_b,
    FP8_MAX: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)

    max_val = 0.0

    start_loc = tl.load(StartLoc + b_id)
    end_loc = tl.load(StartLoc + b_id + 1)
    for t_offset in range(start_loc, end_loc, BLOCK_T):
        t_idx = t_offset + tl.arange(0, BLOCK_T)
        q_range = tl.arange(0, BLOCK_D)
        q_ptrs = Q + t_idx[:, None] * stride_q_t + h_id * stride_q_h + q_range[None, :]
        mask = (t_idx[:, None] < end_loc) & (q_range[None, :] < stride_q_h)
        q_vals = tl.load(q_ptrs, mask=mask, other=0.0)
        max_val = tl.maximum(tl.max(q_vals.abs()), max_val)

    scale = tl.where(max_val > 0, max_val / FP8_MAX, 1.0)
    scale_ptr = Scales + b_id * stride_scales_b + h_id
    tl.store(scale_ptr, scale)


@triton.jit
def _apply_quantization_kernel(
    Q,
    Q_out,
    BatchIds,
    Scales,
    stride_q_t,
    stride_q_h,
    stride_qout_t,
    stride_qout_h,
    stride_scales_b,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t_id = tl.program_id(0)
    h_id = tl.program_id(1)

    batch_id = tl.load(BatchIds + t_id)
    scale_ptr = Scales + batch_id * stride_scales_b + h_id
    scale = tl.load(scale_ptr)

    q_range = tl.arange(0, BLOCK_D)
    q_ptrs = Q + t_id * stride_q_t + h_id * stride_q_h + q_range
    qout_ptrs = Q_out + t_id * stride_qout_t + h_id * stride_qout_h + q_range
    mask = q_range < stride_q_h
    q_vals = tl.load(q_ptrs, mask=mask, other=0.0)
    q_scaled = q_vals / scale
    q_clamped = tl.clamp(q_scaled, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)
    tl.store(qout_ptrs, q_clamped, mask=q_range < stride_qout_h)


@torch.no_grad()
def q_per_head_fp8_quant(q, seq_lens, b1_start_loc, scale_out=None, token_batch_ids=None):
    T, H, D = q.shape
    B = seq_lens.shape[0]

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_T = 256
    num_warps = 4
    num_stages = 2

    q_out = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    if scale_out is None:
        scale_out = torch.empty((B, H), dtype=torch.float32, device=q.device)
    if token_batch_ids is None:
        token_batch_ids = torch.repeat_interleave(torch.arange(B, device=q.device), seq_lens)

    _per_head_max_reduce_kernel[(B, H)](
        q,
        scale_out,
        b1_start_loc,
        q.stride(0),
        q.stride(1),
        scale_out.stride(0),
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _apply_quantization_kernel[(T, H)](
        q,
        q_out,
        token_batch_ids,
        scale_out,
        q.stride(0),
        q.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        scale_out.stride(0),
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return q_out, scale_out


def ref_q_per_head_fp8_quant(q, seq_lens):
    min_fp8 = torch.finfo(torch.float8_e4m3fn).min
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    B = seq_lens.size(0)
    device = q.device
    token_batch_ids = torch.repeat_interleave(torch.arange(B, device=device), seq_lens)
    max_per_time_head = q.abs().amax(dim=2)
    max_per_bh = torch.zeros((B, max_per_time_head.size(1)), device=device, dtype=max_per_time_head.dtype)
    max_per_bh.scatter_reduce_(
        0,
        token_batch_ids.unsqueeze(-1).expand(-1, max_per_time_head.size(1)),
        max_per_time_head,
        reduce="amax",
        include_self=False,
    )
    scales = torch.where(max_per_bh > 0, max_per_bh / max_fp8, torch.ones_like(max_per_bh)).to(torch.float32)
    scale_expanded = scales[token_batch_ids].view(-1, scales.size(1), 1)
    q_q = (q / scale_expanded).clamp(min_fp8, max_fp8).to(torch.float8_e4m3fn)
    return q_q, scales


if __name__ == "__main__":
    B, T, H, D = 200, 1000, 4, 7 * 128
    seq_lens = torch.ones((B,), dtype=torch.int32).cuda() * T // B
    start_locs = torch.zeros(B + 1, dtype=torch.int32).cuda()
    start_locs[1:] = seq_lens.cumsum(dim=0)
    q = torch.randn((T, H, D), dtype=torch.float32).cuda()

    q_out, scales = q_per_head_fp8_quant(q, seq_lens, start_locs)
    q_out1, scales1 = ref_q_per_head_fp8_quant(q, seq_lens)
    assert torch.allclose(scales, scales1, atol=1e-10, rtol=0)
    assert torch.allclose(q_out.int(), q_out1.int(), atol=1e-10, rtol=0)

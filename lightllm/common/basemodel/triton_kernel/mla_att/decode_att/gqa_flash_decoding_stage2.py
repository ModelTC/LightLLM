import os
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage2(
    block_seq_ptr,
    batch_start_index,
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    Out,  # [batch, head, head_dim]
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_obs,
    stride_oh,
    stride_od,
    BLOCK_DMODEL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    cur_head = tl.program_id(0)
    cur_batch = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_index = tl.load(batch_start_index + cur_batch)
    block_seq = tl.load(block_seq_ptr)

    block_n_size = tl.cdiv(cur_batch_seq_len, block_seq)
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_head * stride_mid_oh + cur_batch_start_index * stride_mid_os + offs_d
    offs_logic = cur_head * stride_mid_o_eh + cur_batch_start_index
    for block_seq_n in tl.range(0, block_n_size, 1, num_stages=NUM_STAGES):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)

        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    tl.store(Out + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


@torch.no_grad()
def flash_decode_stage2(
    out_block_seq: torch.Tensor,
    batch_start_index: torch.Tensor,
    mid_out,
    mid_out_logexpsum,
    B_Seqlen,
    Out,
    **run_config
):
    if run_config:
        num_warps = run_config["stage2_num_warps"]
        num_stages = run_config["stage2_num_stages"]

    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128, 256, 512}
    batch, head_num = batch_start_index.shape[0], mid_out.shape[0]
    grid = (head_num, batch)

    _fwd_kernel_flash_decode_stage2[grid](
        out_block_seq,
        batch_start_index,
        B_Seqlen,
        mid_out,
        mid_out_logexpsum,
        Out,
        *mid_out.stride(),
        *mid_out_logexpsum.stride(),
        *Out.stride(),
        BLOCK_DMODEL=Lk,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=1,
    )
    return

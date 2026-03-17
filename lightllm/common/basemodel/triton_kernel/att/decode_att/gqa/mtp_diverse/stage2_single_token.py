"""
MTP Diverse Attention Stage2 Kernel - Single Token Per Request Mode

参考 int8kv diverse stage3 的简化实现：
- 每个请求独立聚合自己的中间结果
- 根据 seq_len 确定需要聚合的 kv block 数量
- 使用 flash attention reweighting 公式
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_mtp_diverse_stage2_single_token(
    B_Seqlen,
    b_mark_shared_group,  # 保留参数用于区分不同的组，但当前实现中不使用
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,  # [batch, num_heads, head_dim]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_ob,
    stride_oh,
    stride_od,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    MTP Diverse Stage2 Kernel - Single Token Per Request Mode

    每个请求独立聚合前 seq_len 个 kv block 的中间结果。
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    # 如果 seq_len 为 0，直接返回
    if cur_batch_seq_len == 0:
        return

    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 计算需要处理的 kv block 数量
    block_n_size = tl.cdiv(cur_batch_seq_len, BLOCK_SEQ)

    # 初始化 accumulator
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for block_idx in range(0, block_n_size, 1):
        # 加载第 block_idx 个 kv block 的中间结果
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + block_idx * stride_mid_os
            + offs_d[:]
        )
        offs_mid_o_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh + block_idx

        mid_o_val = tl.load(Mid_O + offs_mid_o)
        logic_val = tl.load(Mid_O_LogExpSum + offs_mid_o_logic)

        # Flash attention reweighting
        new_max_logic = tl.maximum(logic_val, max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        exp_val = tl.exp(logic_val - new_max_logic)

        acc = acc * logic_scale + exp_val * mid_o_val
        sum_exp = sum_exp * logic_scale + exp_val
        max_logic = new_max_logic

    # 归一化并存储结果
    offs_o = cur_batch * stride_ob + cur_head * stride_oh + offs_d
    tl.store(O + offs_o, acc / sum_exp)

    return


@torch.no_grad()
def mtp_diverse_stage2_single_token(
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    B_Seqlen: torch.Tensor,
    b_mark_shared_group: torch.Tensor,
    O: torch.Tensor,
    block_seq: int,
):
    """
    MTP Diverse Attention Stage2 - Single Token Per Request Mode

    参数：
    - mid_out: [batch, head, seq_block_num, head_dim] - 中间结果
    - mid_out_logsumexp: [batch, head, seq_block_num] - 中间 logsumexp
    - B_Seqlen: [batch] - 每个请求的 seq_len
    - b_mark_shared_group: [batch] - 组标记（保留参数，当前未使用）
    - O: [batch, num_heads, head_dim] - 输出
    - block_seq: 块大小
    """
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)

    _fwd_kernel_mtp_diverse_stage2_single_token[grid](
        B_Seqlen=B_Seqlen,
        b_mark_shared_group=b_mark_shared_group,
        Mid_O=mid_out,
        Mid_O_LogExpSum=mid_out_logsumexp,
        O=O,
        stride_mid_ob=mid_out.stride(0),
        stride_mid_oh=mid_out.stride(1),
        stride_mid_os=mid_out.stride(2),
        stride_mid_od=mid_out.stride(3),
        stride_mid_o_eb=mid_out_logsumexp.stride(0),
        stride_mid_o_eh=mid_out_logsumexp.stride(1),
        stride_mid_o_es=mid_out_logsumexp.stride(2),
        stride_ob=O.stride(0),
        stride_oh=O.stride(1),
        stride_od=O.stride(2),
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=Lk,
        num_warps=4,
        num_stages=2,
    )
    return

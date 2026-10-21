import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_normal_decode_stage2(
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    O,  # [batch, head, head_dim]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_obs,
    stride_oh,
    stride_od,
    block_num,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_num = tl.minimum(tl.cdiv(cur_batch_seq_len, BLOCK_SEQ), block_num)

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_index in range(0, block_num, 1):
        tv = tl.load(Mid_O + offs_v + block_index * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_index)
        new_max_logic = tl.maximum(tlogic, max_logic)

        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


@torch.no_grad()
def flash_decode_stage2(
    mid_out: torch.Tensor,
    mid_out_logexpsum: torch.Tensor,
    B_Seqlen: torch.Tensor,
    O: torch.Tensor,
    block_seq: int,
):
    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128, 256}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)
    block_num = mid_out.shape[2]

    _fwd_kernel_flash_normal_decode_stage2[grid](
        B_Seqlen=B_Seqlen,
        Mid_O=mid_out,
        Mid_O_LogExpSum=mid_out_logexpsum,
        O=O,
        stride_mid_ob=mid_out.stride(0),
        stride_mid_oh=mid_out.stride(1),
        stride_mid_os=mid_out.stride(2),
        stride_mid_od=mid_out.stride(3),
        stride_mid_o_eb=mid_out_logexpsum.stride(0),
        stride_mid_o_eh=mid_out_logexpsum.stride(1),
        stride_mid_o_es=mid_out_logexpsum.stride(2),
        stride_obs=O.stride(0),
        stride_oh=O.stride(1),
        stride_od=O.stride(2),
        block_num=block_num,
        BLOCK_SEQ=block_seq,
        BLOCK_DMODEL=Lk,
        num_warps=4,
        num_stages=2,
    )
    return

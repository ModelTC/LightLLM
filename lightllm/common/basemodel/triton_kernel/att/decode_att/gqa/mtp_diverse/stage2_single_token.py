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
from typing import Optional
from lightllm.common.triton_utils.autotuner import autotune, Autotuner, AutotuneKernelType, AutotuneLevel
from lightllm.utils.envs_utils import get_decode_attn_autotune_seq_len, get_triton_autotune_level


def get_test_configs():
    configs = []
    for num_warps in [2, 4, 8, 16]:
        for num_stages in [1, 2, 4, 5, 6]:
            configs.append(
                {
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                }
            )
    return configs


def get_static_key(mid_out, block_n, out):
    key_params = {
        "q_head_dim": int(mid_out.shape[-1]),
        "block_n": block_n,
        "out_dtype": str(out.dtype),
    }
    return key_params


def get_run_key(mid_out, block_n, max_kv_len):
    batch_size_head = mid_out.shape[0] * mid_out.shape[1]
    # 正常查找使用外部保存的真实 KV 长度；只有 decode 调优时才使用重建输入的目标长度。
    max_kv_len = int(max_kv_len)
    if Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION) and get_triton_autotune_level() in [
        AutotuneLevel.ADAPTIVE_AUTOTUNE,
        AutotuneLevel.FORCE_AUTOTUNE,
    ]:
        max_kv_len = get_decode_attn_autotune_seq_len()
    # stage2 只归约 stage1 输出的有效分块，工作量达到缓冲区分块数后不再随 KV 长度增长。
    # 直接以有效分块数区分配置，不再按 512 token 分桶，以免短请求的 key 与实际归约次数不符。
    block_num = min(triton.cdiv(max_kv_len, block_n), mid_out.shape[2])
    return batch_size_head * 1000 * 1000 * 1000 + block_num


def rebuild_inputs(
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    B_Seqlen: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    max_kv_len: int,
    **kwargs,
):
    # 仅在实际搜索前构造一次输入，不计入 benchmark；正式执行和 Graph 捕获仍使用原始中间结果。
    max_kv_len = get_decode_attn_autotune_seq_len()
    # stage2 按行独立归约，不需要共享组标记或 KV 映射，以目标长度代表各行的归约工作量。
    B_Seqlen = torch.full_like(B_Seqlen, max_kv_len, device="cpu").to(device=B_Seqlen.device)

    # mid_out 和 mid_out_logsumexp 是 stage1 产出的只读输入，stage2 不会修改它们。
    # 两个张量的 shape 已按最大分块数分配，调优时直接复用原始输入，避免无意义的重复分配。
    # block_n 必须沿用 stage1 实际返回的 BLOCK_N，它决定每行需要归约的分块数，不能独立改写。
    # benchmark 的最终输出写入继续由 mutates_args 隔离，原始 out 不会被调优覆盖。
    return (mid_out, mid_out_logsumexp, B_Seqlen, out, block_n, max_kv_len), kwargs


@triton.jit
def _fwd_kernel_mtp_diverse_stage2_single_token(
    B_Seqlen,
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
    mid_out_block_num,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    MTP Diverse Stage2 Kernel - Single Token Per Request Mode

    每个请求独立聚合前 seq_len 个 kv block 的中间结果。
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DMODEL)

    # 计算需要处理的 kv block 数量
    block_n_size = tl.cdiv(cur_batch_seq_len, BLOCK_N)
    block_n_size = tl.minimum(block_n_size, mid_out_block_num)

    # 初始化 accumulator
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for block_idx in tl.range(0, block_n_size, 1, num_stages=NUM_STAGES):
        # 加载第 block_idx 个 kv block 的中间结果
        offs_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + block_idx * stride_mid_os + offs_d[:]
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


@autotune(
    kernel_name="_fwd_kernel_mtp_diverse_stage2_single_token:v3",
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    configs_gen_func=get_test_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    rebuild_input_func=rebuild_inputs,
    mutates_args=["out"],
)
@torch.no_grad()
def mtp_diverse_stage2_single_token(
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    B_Seqlen: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    max_kv_len: int,
    run_config: Optional[dict] = None,
):
    if not run_config:
        run_config = {"num_warps": 4, "num_stages": 2}

    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]

    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (batch, head_num)

    _fwd_kernel_mtp_diverse_stage2_single_token[grid](
        B_Seqlen=B_Seqlen,
        Mid_O=mid_out,
        Mid_O_LogExpSum=mid_out_logsumexp,
        O=out,
        stride_mid_ob=mid_out.stride(0),
        stride_mid_oh=mid_out.stride(1),
        stride_mid_os=mid_out.stride(2),
        stride_mid_od=mid_out.stride(3),
        stride_mid_o_eb=mid_out_logsumexp.stride(0),
        stride_mid_o_eh=mid_out_logsumexp.stride(1),
        stride_mid_o_es=mid_out_logsumexp.stride(2),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_od=out.stride(2),
        mid_out_block_num=mid_out.shape[2],
        BLOCK_N=block_n,
        BLOCK_DMODEL=Lk,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return


if __name__ == "__main__":
    if get_triton_autotune_level() != 2:
        raise Exception("you need set env LIGHTLLM_TRITON_AUTOTUNE_LEVEL=2 to start program.")

    q_head_dim = 128
    tp_world_size = 2
    out_dtype = torch.float

    batch_sizes = [1, 8, 16, 32, 64, 128]
    q_head_num = 64 // tp_world_size
    max_kv_len = get_decode_attn_autotune_seq_len()

    Autotuner.start_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION)
    # autotuing kernel
    for batch_size in batch_sizes:
        for block_n in [16, 32, 64, 128]:
            for out_dtype in [torch.float16, torch.bfloat16]:
                if batch_size <= 16:
                    block_num = 128
                elif batch_size <= 64:
                    block_num = 64
                else:
                    block_num = 32

                mid_out = torch.randn(
                    batch_size, q_head_num, block_num, q_head_dim, dtype=torch.bfloat16, device="cuda"
                )
                mid_out_logsumexp = torch.randn(batch_size, q_head_num, block_num, dtype=torch.float32, device="cuda")
                B_Seqlen = torch.full((batch_size,), max_kv_len, dtype=torch.int32, device="cuda")
                out = torch.zeros(batch_size, q_head_num, q_head_dim, dtype=out_dtype, device="cuda")

                mtp_diverse_stage2_single_token(
                    mid_out=mid_out,
                    mid_out_logsumexp=mid_out_logsumexp,
                    B_Seqlen=B_Seqlen,
                    out=out,
                    block_n=block_n,
                    max_kv_len=max_kv_len,
                )

    Autotuner.end_autotune_warmup()

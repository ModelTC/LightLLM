import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F

from lightllm.utils.device_utils import get_device_sm_count
from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _fwd_kernel(
    Q_nope,
    Q_rope,
    KV_nope,
    KV_rope,
    q_nope_dim: tl.constexpr,
    q_rope_dim: tl.constexpr,
    qhead: tl.constexpr,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,
    Out,
    Req_to_tokens,
    B_req_idx,
    batch_size: tl.constexpr,
    mtp_size: tl.constexpr,
    stride_qnope_bs,
    stride_qnope_h,
    stride_qrope_bs,
    stride_qrope_h,
    stride_kvnope_bs,
    stride_kvrope_bs,
    stride_o_bs,
    stride_req_to_token_b,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_count: tl.constexpr,
    FLATTEN: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # 优化提示：告诉编译器维度是16的倍数

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # q
    offs_d = tl.arange(0, q_nope_dim)
    offs_dv = tl.arange(0, q_rope_dim)
    offs_md = offs_m[:, None] * stride_qnope_h + offs_d[None, :]
    offs_mdv = offs_m[:, None] * stride_qrope_h + offs_dv[None, :]

    for cur_batch in range(pid, batch_size, sm_count):
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
        nblock_num = tl.cdiv(cur_batch_seq_len, BLOCK_N)

        offs_q = cur_batch_in_all_start_index * stride_qnope_bs + offs_md
        offs_qv = cur_batch_in_all_start_index * stride_qrope_bs + offs_mdv

        q_nope = tl.load(Q_nope + offs_q)
        q_rope = tl.load(Q_rope + offs_qv)
        req_to_tokens_ptr = Req_to_tokens + stride_req_to_token_b * cur_batch_req_idx

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, q_nope_dim], dtype=tl.float32)

        # 从后往前遍历KV，只有第一个block需要causal mask
        first_iter = True
        for block_idx in tl.range(nblock_num - 1, -1, -1, flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE):
            start_n = block_idx * BLOCK_N

            offs_n_new = start_n + offs_n
            seq_n_mask = offs_n_new < cur_batch_seq_len
            seq_n_mask_2d = seq_n_mask[None, :]
            kv_loc = tl.load(
                req_to_tokens_ptr + offs_n_new,
                mask=seq_n_mask,
                other=0,
            ).to(tl.int64)
            offs_kv = kv_loc[None, :] * stride_kvnope_bs + offs_d[:, None]
            kv_nope = tl.load(KV_nope + offs_kv, mask=seq_n_mask_2d, other=0.0)
            offs_kv_rope = kv_loc[None, :] * stride_kvrope_bs + offs_dv[:, None]
            kv_rope = tl.load(KV_rope + offs_kv_rope, mask=seq_n_mask_2d, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            qk = tl.dot(q_nope, kv_nope)
            qk += tl.dot(q_rope, kv_rope)
            qk *= sm_scale

            # MTP causal mask: 只在第一次循环（最后一个block）需要mask
            if first_iter:
                # token_idx: 第i个token在BLOCK_M中的索引 [0, 0, ..., 1, 1, ..., mtp_size-1]
                token_idx = offs_m // qhead  # [BLOCK_M]
                # 每个token的最大可见位置: cur_batch_seq_len - mtp_size + 1 + token_idx
                max_visible_pos = cur_batch_seq_len - mtp_size + 1 + token_idx  # [BLOCK_M]
                # 生成阶梯状causal mask
                mask = offs_n_new[None, :] < max_visible_pos[:, None]  # [BLOCK_M, BLOCK_N]
                qk = tl.where(mask, qk, float("-inf"))

            # max_get_scale
            m_ij = tl.max(qk, 1)
            if first_iter:
                m_i = m_ij
                # 处理 m_i = -inf：避免 exp(-inf - (-inf)) = NaN
                qk_safe = tl.where(m_i[:, None] == float("-inf"), float("-inf"), qk - m_i[:, None])
                qk = tl.exp(qk_safe)
                acc = tl.dot(qk.to(kv_nope.dtype), tl.trans(kv_nope))
                l_ij = tl.sum(qk, 1)
                l_i = l_ij
            else:
                m_i_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_i_new)
                m_i = m_i_new
                qk = tl.exp(qk - m_i[:, None])
                acc = acc * alpha[:, None] + tl.dot(qk.to(kv_nope.dtype), tl.trans(kv_nope))
                l_ij = tl.sum(qk, 1)
                l_i = alpha * l_i + l_ij

            first_iter = False

        inv_l_i = 1.0 / l_i
        acc = acc * inv_l_i[:, None]
        offs_o = cur_batch_in_all_start_index * stride_o_bs + offs_md
        tl.store(Out + offs_o, acc)

    return


def get_fa3_mla_mtp_configs():
    """生成候选配置列表"""
    configs = []
    # BLOCK_N 候选值
    block_n_list = [32, 64, 128]
    # num_warps 候选值
    num_warps_list = [4, 8, 16]
    # num_stages 候选值
    num_stages_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # flatten 候选值
    flatten_list = [True, False]
    # warp_specialize 候选值
    warp_specialize_list = [True, False]

    for BLOCK_N in block_n_list:
        for num_warps in num_warps_list:
            for num_stages in num_stages_list:
                for flatten in flatten_list:
                    for warp_specialize in warp_specialize_list:
                        configs.append(
                            {
                                "BLOCK_N": BLOCK_N,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                                "flatten": flatten,
                                "warp_specialize": warp_specialize,
                            }
                        )
    return configs


def static_key_func(q_nope, q_rope, kv_nope, kv_rope, mtp_size):
    """静态key：标识kernel的类型特征，用于确定缓存文件"""
    return {
        "dtype": str(q_nope.dtype),
        "q_nope_dim": q_nope.shape[-1],
        "q_rope_dim": q_rope.shape[-1],
        "qhead": q_nope.shape[1],
        "mtp_size": mtp_size,
    }


def run_key_func(b_seq_len):
    """运行key：标识具体的运行场景，用于索引配置"""
    # 使用平均序列长度作为 run_key
    avg_seq_len = int(b_seq_len.float().mean().item())
    return avg_seq_len


@autotune(
    kernel_name="fa3_mla_mtp",
    configs_gen_func=get_fa3_mla_mtp_configs,
    static_key_func=static_key_func,
    run_key_func=run_key_func,
    mutates_args=["o"],  # o 会被修改
)
@torch.no_grad()
def fa3_mla_mtp(
    q_nope,
    q_rope,
    kv_nope,
    kv_rope,
    o,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    mtp_size,
    req_to_token_indexs,
    run_config=None,
):

    q_nope_dim = q_nope.shape[-1]
    q_rope_dim = q_rope.shape[-1]
    assert q_nope_dim == kv_nope.shape[-1]
    assert q_rope_dim == kv_rope.shape[-1]

    batch_size, qhead = b_seq_len.shape[0], q_nope.shape[1]

    # 从 run_config 获取配置，或使用默认值
    if run_config is None:
        run_config = {}
    BLOCK_N = run_config.get("BLOCK_N", 64)
    num_warps = run_config.get("num_warps", 8)
    num_stages = run_config.get("num_stages", 1)
    flatten = run_config.get("flatten", False)
    warp_specialize = run_config.get("warp_specialize", False)

    BLOCK_M = mtp_size * qhead

    sm_count = get_device_sm_count()
    num_blocks_m = triton.cdiv(batch_size * qhead * mtp_size, BLOCK_M)
    grid = (min(sm_count, num_blocks_m),)
    softmax_scale = 1.0 / math.sqrt(q_nope_dim + q_rope_dim)

    _fwd_kernel[grid](
        q_nope,
        q_rope,
        kv_nope,
        kv_rope,
        q_nope_dim,
        q_rope_dim,
        qhead,
        softmax_scale,
        b_start_loc,
        b_seq_len,
        o,
        req_to_token_indexs,
        b_req_idx,
        batch_size,
        mtp_size,
        q_nope.stride(0),  # stride_qnope_bs
        q_nope.stride(1),  # stride_qnope_h
        q_rope.stride(0),  # stride_qrope_bs
        q_rope.stride(1),  # stride_qrope_h
        kv_nope.stride(0),  # stride_kvnope_bs
        kv_rope.stride(0),  # stride_kvrope_bs
        o.stride(0),  # stride_o_bs
        req_to_token_indexs.stride(0),  # stride_req_to_token_b
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        sm_count=sm_count,
        FLATTEN=flatten,
        WARP_SPECIALIZE=warp_specialize,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return


def test():
    import torch
    import numpy as np

    Z, N_CTX, H, S_Q, D_HEAD, ROPE_HEAD = 128, 8192, 16, 2, 512, 64
    prompt_cache_len = N_CTX - S_Q
    dtype = torch.float16

    q = torch.empty((Z * S_Q, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    q_rope = torch.empty((Z * S_Q, H, ROPE_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    kv = torch.empty((Z * N_CTX, 1, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    kv_rope = torch.empty((Z * N_CTX, 1, ROPE_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    o = torch.empty((Z * S_Q, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.7, std=0.2)

    req_to_token_indexs = torch.zeros((1000, N_CTX + 10), dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_req_idx = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(Z, dtype=torch.int32, device="cuda")

    for i in range(Z):
        b_seq_len[i] = N_CTX
        b_req_idx[i] = i
        req_to_token_indexs[i][:N_CTX] = torch.tensor(np.arange(N_CTX), dtype=torch.int32).cuda() + N_CTX * i
        if i != 0:
            b_start_loc[i] = b_start_loc[i - 1] + N_CTX - prompt_cache_len
        b_prompt_cache_len[i] = prompt_cache_len

    fa3_mla_mtp(
        q,
        q_rope,
        kv,
        kv_rope,
        o,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        S_Q,
        req_to_token_indexs,
    )


def test_with_autotune():
    """使用自动调优的测试"""
    import torch
    import numpy as np
    import os
    from lightllm.common.triton_utils.autotuner import Autotuner

    # 设置自动调优级别
    # 0: 使用缓存配置 (USE_AUTOTUNE_HIS_CONFIG)
    # 1: 缺少配置时自动调优 (ADAPTIVE_AUTOTUNE)
    # 2: 强制重新调优 (FORCE_AUTOTUNE)
    # 3: 关闭自动调优 (CLOSE_AUTOTUNE)
    os.environ["LIGHTLLM_TRITON_AUTOTUNE_LEVEL"] = "1"  # ADAPTIVE_AUTOTUNE

    # 设置详细输出模式
    # 0: 只显示进度条和最佳时间（默认）
    # 1: 显示每个配置的测试时长和详细结果
    os.environ["LIGHTLLM_TRITON_AUTOTUNE_VERBOSE"] = "1"  # 启用详细输出

    Z, N_CTX, H, S_Q, D_HEAD, ROPE_HEAD = 128, 8192, 16, 2, 512, 64
    prompt_cache_len = N_CTX - S_Q
    dtype = torch.float16

    q = torch.empty((Z * S_Q, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    q_rope = torch.empty((Z * S_Q, H, ROPE_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    kv = torch.empty((Z * N_CTX, 1, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)
    kv_rope = torch.empty((Z * N_CTX, 1, ROPE_HEAD), dtype=dtype, device="cuda").normal_(mean=0.3, std=0.2)

    o = torch.empty((Z * S_Q, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.7, std=0.2)

    req_to_token_indexs = torch.zeros((1000, N_CTX + 10), dtype=torch.int32, device="cuda")
    b_start_loc = torch.zeros((Z,), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_req_idx = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_prompt_cache_len = torch.zeros(Z, dtype=torch.int32, device="cuda")

    for i in range(Z):
        b_seq_len[i] = N_CTX
        b_req_idx[i] = i
        req_to_token_indexs[i][:N_CTX] = torch.tensor(np.arange(N_CTX), dtype=torch.int32).cuda() + N_CTX * i
        if i != 0:
            b_start_loc[i] = b_start_loc[i - 1] + N_CTX - prompt_cache_len
        b_prompt_cache_len[i] = prompt_cache_len

    print("🚀 开始自动调优...")
    # 启动 warmup 模式才会触发自动调优
    Autotuner.start_autotune_warmup()

    fa3_mla_mtp(
        q,
        q_rope,
        kv,
        kv_rope,
        o,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        S_Q,
        req_to_token_indexs,
    )

    Autotuner.end_autotune_warmup()
    print("✅ 自动调优完成！配置已缓存。")

    # 后续调用将使用最优配置
    print("🔥 使用最优配置运行...")
    for _ in range(3):
        fa3_mla_mtp(
            q,
            q_rope,
            kv,
            kv_rope,
            o,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            S_Q,
            req_to_token_indexs,
        )
    print("✅ 完成！")


if __name__ == "__main__":
    # 使用 test() 进行基础测试
    # test()

    # 使用 test_with_autotune() 进行自动调优
    test_with_autotune()

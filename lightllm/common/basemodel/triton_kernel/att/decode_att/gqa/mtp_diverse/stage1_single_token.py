"""
MTP Diverse Attention Stage1 Kernel - Single Token Per Request Mode

简化版本（参考 int8kv diverse stage1）：
- 组内请求 [q1, q2, q3, q4]，group mark [0, 0, 0, 4]
- KV slots [kv0, kv1, kv2, kv3, kv4]
- 可见性：q1->[kv0], q2->[kv0,kv1], q3->[kv0,kv1,kv2], q4->[kv0,kv1,kv2,kv3,kv4]

核心逻辑：
- 只由组内最后一个请求（b_mark_shared_group != 0）触发计算
- 一次加载组内所有请求的 Q 和 KV
- 每个请求单独做可见性检查（基于各自的 seq_len）
- 中间结果按 kv block 存储，供 Stage2 聚合
"""
import torch
import triton
import triton.language as tl
from typing import Optional
from lightllm.common.triton_utils.autotuner import autotune, Autotuner


def get_test_configs():
    configs = []
    for block_n in [16, 32, 64, 128]:
        for num_warps in [2, 4, 8, 16]:
            for num_stages in [2, 4, 6]:
                configs.append(
                    {
                        "BLOCK_N": block_n,
                        "num_warps": num_warps,
                        "num_stages": num_stages,
                    }
                )
    return configs


def get_static_key(q, k, block_seq):
    key_params = {
        "gqa_group_size": int(q.shape[1] // k.shape[1]),
        "q_head_dim": int(q.shape[2]),
        "block_seq": block_seq,
        "out_dtype": str(q.dtype),
    }
    return key_params


def get_run_key(q, max_kv_len):
    batch_size = q.shape[0]
    return batch_size * 1000 * 1000 * 1000 + max_kv_len


@triton.jit
def _fwd_kernel_mtp_diverse_stage1_single_token(
    Q,
    stride_qb,
    stride_qh,
    stride_qd,
    K,
    stride_kbs,
    stride_kh,
    stride_kd,
    V,
    stride_vbs,
    stride_vh,
    stride_vd,
    sm_scale,
    Req_to_tokens,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    B_req_idx,
    b_seq_len,
    b_mark_shared_group,
    Mid_O,  # [batch, head, seq_block_num, head_dim] - 每个 kv block 的中间结果
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    MTP Diverse Stage1 Kernel - Single Token Per Request Mode

    使用 BLOCK_BATCH 同时处理组内所有请求，一次加载 KV 服务多个 Q。
    只由组内最后一个请求（b_mark_shared_group != 0）触发计算。
    中间结果按 kv block 存储，供 Stage2 聚合。
    """
    cur_batch = tl.program_id(0)
    shared_batch_group_size = tl.load(b_mark_shared_group + cur_batch)

    # 如果不是组内最后一个请求，跳过
    if shared_batch_group_size == 0:
        return

    # 计算组的起始位置
    cur_batch_end = cur_batch + 1
    cur_batch_start = cur_batch - (shared_batch_group_size - 1)

    cur_kv_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    # 使用固定的 BLOCK_BATCH = 16，避免 CUDA graph 动态形状问题
    BLOCK_BATCH: tl.constexpr = 16

    # Q head range - 使用实际的 gqa_group_size 作为 BLOCK_HEAD 的上限
    cur_q_head_range = cur_kv_head * gqa_group_size + tl.arange(0, BLOCK_HEAD)
    q_head_end_index = (cur_kv_head + 1) * gqa_group_size
    cur_q_head_range = tl.where(cur_q_head_range < q_head_end_index, cur_q_head_range, cur_kv_head * gqa_group_size)

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # 加载组内所有请求的 b_seq_len 和 b_req_idx
    offs_batch = cur_batch_start + tl.arange(0, BLOCK_BATCH)
    offs_batch = tl.where(offs_batch < cur_batch_end, offs_batch, cur_batch_start)

    batch_seq_lens = tl.load(b_seq_len + offs_batch)  # [BLOCK_BATCH] - 每个请求的 seq_len
    batch_req_idxs = tl.load(B_req_idx + offs_batch)  # [BLOCK_BATCH] - 每个请求的 req_idx

    # 当前 block 处理的 KV 范围 [kv_start, kv_end)
    kv_start = seq_start_block * BLOCK_SEQ
    # 使用组内最大的 seq_len 来决定需要处理多少 KV
    max_seq_len = tl.max(batch_seq_lens)
    kv_end = tl.minimum(max_seq_len, kv_start + BLOCK_SEQ)

    off_q = offs_batch[:, None, None] * stride_qb + cur_q_head_range[None, :, None] * stride_qh + offs_d[None, None, :]

    block_n_size = tl.cdiv(
        tl.where(kv_end - kv_start <= 0, 0, kv_end - kv_start),
        BLOCK_N,
    )

    if block_n_size == 0:
        return

    offs_n = kv_start + tl.arange(0, BLOCK_N)

    # 加载 Q - 保持 3D 形状用于 broadcasting
    q = tl.load(Q + off_q)  # [BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM]

    # 初始化 accumulator - 保持 3D 形状
    sum_exp = tl.zeros([BLOCK_BATCH, BLOCK_HEAD], dtype=tl.float32)
    max_logic = tl.zeros([BLOCK_BATCH, BLOCK_HEAD], dtype=tl.float32) - float("inf")
    acc = tl.zeros([BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM], dtype=tl.float32)

    # 使用组内最后一个请求的 KV 索引（因为它的 seq_len 最大，所有位置都有效）
    # 组内最后一个请求的索引是 cur_batch（因为 cur_batch_end = cur_batch + 1）
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        n_mask = offs_n_new < kv_end

        # 加载 KV 索引（使用组内最后一个请求的索引，因为它有最大的 seq_len）
        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=n_mask,
            other=0,
        ).to(tl.int64)

        # 加载 K: [BLOCK_N, BLOCK_HEADDIM]
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
        k = tl.load(K + off_k, mask=n_mask[:, None], other=0.0)

        # 加载 V: [BLOCK_N, BLOCK_HEADDIM]
        off_v = k_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
        v = tl.load(V + off_v, mask=n_mask[:, None], other=0.0)

        # QK^T: [BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM] @ [BLOCK_N, BLOCK_HEADDIM]^T
        # 使用 tl.dot 需要 reshape
        q_flat = tl.reshape(q, (BLOCK_BATCH * BLOCK_HEAD, BLOCK_HEADDIM))
        k_flat = tl.reshape(k, (BLOCK_N, BLOCK_HEADDIM))
        att_value_flat = tl.dot(q_flat, tl.trans(k_flat))  # [BLOCK_BATCH * BLOCK_HEAD, BLOCK_N]
        att_value = tl.reshape(att_value_flat, (BLOCK_BATCH, BLOCK_HEAD, BLOCK_N))
        att_value *= sm_scale

        # 每个请求有自己的可见性检查
        # batch_seq_lens: [BLOCK_BATCH], offs_n_new: [BLOCK_N]
        # visible_mask: [BLOCK_BATCH, BLOCK_N] - 每个请求对每个 KV 位置的可见性
        kv_positions = offs_n_new[None, :]  # [1, BLOCK_N]
        batch_seq_lens_expanded = batch_seq_lens[:, None]  # [BLOCK_BATCH, 1]
        visible_mask = kv_positions < batch_seq_lens_expanded  # [BLOCK_BATCH, BLOCK_N]

        # combined_mask: 同时考虑 KV 边界和 seq_len 可见性
        combined_mask = n_mask[None, :] & visible_mask
        combined_mask_expanded = combined_mask[:, None, :]  # [BLOCK_BATCH, 1, BLOCK_N]

        att_value = tl.where(combined_mask_expanded, att_value, float("-inf"))

        # Flash attention update - 3D 版本
        cur_max_logic = tl.max(att_value, axis=2)  # [BLOCK_BATCH, BLOCK_HEAD]
        new_max_logic = tl.maximum(cur_max_logic, max_logic)

        exp_logic = tl.exp(att_value - new_max_logic[:, :, None])  # [BLOCK_BATCH, BLOCK_HEAD, BLOCK_N]
        logic_scale = tl.exp(max_logic - new_max_logic)  # [BLOCK_BATCH, BLOCK_HEAD]
        acc *= logic_scale[:, :, None]  # [BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM]

        # tl.dot 需要 2D @ 2D
        exp_logic_flat = tl.reshape(exp_logic, (BLOCK_BATCH * BLOCK_HEAD, BLOCK_N))
        acc_flat = tl.reshape(acc, (BLOCK_BATCH * BLOCK_HEAD, BLOCK_HEADDIM))
        acc_flat += tl.dot(exp_logic_flat.to(q.dtype), v)
        acc = tl.reshape(acc_flat, (BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM))

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=2)  # [BLOCK_BATCH, BLOCK_HEAD]
        max_logic = new_max_logic

    # 存储中间结果：每个 kv block 存储一次，包含组内所有请求的结果
    off_mid_o = (
        offs_batch[:, None, None] * stride_mid_ob
        + cur_q_head_range[None, :, None] * stride_mid_oh
        + seq_start_block * stride_mid_os
        + offs_d[None, None, :]
    )
    off_mid_o_logexpsum = (
        offs_batch[:, None] * stride_mid_o_eb + cur_q_head_range[None, :] * stride_mid_o_eh + seq_start_block
    )

    # 归一化并存储
    mid_o_val = acc / sum_exp[:, :, None]
    mid_logic_val = max_logic + tl.log(sum_exp)

    tl.store(
        Mid_O + off_mid_o,
        mid_o_val,
    )
    tl.store(
        Mid_O_LogExpSum + off_mid_o_logexpsum,
        mid_logic_val,
    )

    return


@autotune(
    kernel_name="_fwd_kernel_mtp_diverse_stage1_single_token:v1",
    configs_gen_func=get_test_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    mutates_args=["mid_out", "mid_out_logsumexp"],
)
@torch.no_grad()
def mtp_diverse_stage1_single_token(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_mark_shared_group: torch.Tensor,
    max_kv_len: int,
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    block_seq: int,
    run_config: Optional[dict] = None,
):
    """
    MTP Diverse Attention Stage1 - Single Token Per Request Mode

    b_seq_len: 每个请求可见的 KV 数量，组内递增
    例如组内 [q1, q2, q3, q4] 对应 b_seq_len [2, 3, 4, 5]
    """
    if not run_config:
        run_config = {"BLOCK_N": 16, "num_warps": 2, "num_stages": 2}

    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]

    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    BLOCK_SEQ = block_seq
    assert BLOCK_SEQ % BLOCK_N == 0
    Lq, Lk = q.shape[2], k.shape[2]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    sm_scale = 1.0 / (Lk**0.5)
    batch, kv_head_num = B_req_idx.shape[0], k.shape[1]
    grid = (batch, kv_head_num, triton.cdiv(max_kv_len, BLOCK_SEQ))
    gqa_group_size = q.shape[1] // k.shape[1]
    assert triton.next_power_of_2(Lk) == Lk

    # BLOCK_HEAD 设置为 gqa_group_size 向上取整到 2 的幂
    BLOCK_HEAD = triton.next_power_of_2(gqa_group_size)
    # 使用固定的 BLOCK_BATCH = 16，避免 CUDA graph 捕获时的动态形状问题
    BLOCK_BATCH = 16

    assert k.stride() == v.stride()

    _fwd_kernel_mtp_diverse_stage1_single_token[grid](
        Q=q,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        K=k,
        stride_kbs=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        V=v,
        stride_vbs=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        sm_scale=sm_scale,
        Req_to_tokens=Req_to_tokens,
        stride_req_to_tokens_b=Req_to_tokens.stride(0),
        stride_req_to_tokens_s=Req_to_tokens.stride(1),
        B_req_idx=B_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        Mid_O=mid_out,
        stride_mid_ob=mid_out.stride(0),
        stride_mid_oh=mid_out.stride(1),
        stride_mid_os=mid_out.stride(2),
        stride_mid_od=mid_out.stride(3),
        Mid_O_LogExpSum=mid_out_logsumexp,
        stride_mid_o_eb=mid_out_logsumexp.stride(0),
        stride_mid_o_eh=mid_out_logsumexp.stride(1),
        stride_mid_o_es=mid_out_logsumexp.stride(2),
        gqa_group_size=gqa_group_size,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=block_seq,
        BLOCK_HEADDIM=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return


if __name__ == "__main__":
    from lightllm.utils.envs_utils import get_triton_autotune_level

    if get_triton_autotune_level() != 2:
        raise Exception("you need set env LIGHTLLM_TRITON_AUTOTUNE_LEVEL=2 to start program.")

    # static params
    gqa_group_size = 4
    q_head_dim = 128
    block_seq = 128
    out_dtype = torch.bfloat16

    batch_sizes = [1, 8, 16, 32, 64, 128]
    decode_lengths = [1024, 2048, 8192, 16384]

    q_head_num = gqa_group_size

    Autotuner.start_autotune_warmup()
    # autotuing kernel
    for batch_size in batch_sizes:
        for length in decode_lengths:
            # Setup test tensors
            q = torch.randn(batch_size, q_head_num, q_head_dim, dtype=out_dtype, device="cuda")
            k = torch.randn(batch_size * length, 1, q_head_dim, dtype=out_dtype, device="cuda")
            v = torch.randn(batch_size * length, 1, q_head_dim, dtype=out_dtype, device="cuda")
            Req_to_tokens = torch.arange(0, batch_size * length, dtype=torch.int32, device="cuda").view(
                batch_size, length
            )
            B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
            B_seq_len = torch.full((batch_size,), length, dtype=torch.int32, device="cuda")
            b_mark_shared_group = torch.arange(1, batch_size + 1, dtype=torch.int32, device="cuda")

            if batch_size <= 16:
                block_num = 128
            elif batch_size <= 64:
                block_num = 64
            else:
                block_num = 32

            mid_out = torch.zeros(batch_size, q_head_num, block_num, q_head_dim, dtype=out_dtype, device="cuda")
            mid_out_logsumexp = torch.zeros(batch_size, q_head_num, block_num, dtype=out_dtype, device="cuda")

            mtp_diverse_stage1_single_token(
                q=q,
                k=k,
                v=v,
                Req_to_tokens=Req_to_tokens,
                B_req_idx=B_req_idx,
                b_seq_len=B_seq_len,
                b_mark_shared_group=b_mark_shared_group,
                max_kv_len=length,
                mid_out=mid_out,
                mid_out_logsumexp=mid_out_logsumexp,
                block_seq=block_seq,
            )

    Autotuner.end_autotune_warmup()

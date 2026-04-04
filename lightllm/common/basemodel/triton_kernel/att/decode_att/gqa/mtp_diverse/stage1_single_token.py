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
from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size


def get_test_configs():
    configs = []
    for block_n in [16, 32, 64]:
        for num_warps in [2, 4, 8]:
            for num_stages in [2, 3, 4]:
                configs.append({
                    "BLOCK_N": block_n,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                })
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
    stride_qb, stride_qh, stride_qd,
    K,
    stride_kbs, stride_kh, stride_kd,
    V,
    stride_vbs, stride_vh, stride_vd,
    sm_scale,
    Req_to_tokens,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    B_req_idx,
    b_seq_len,
    b_mark_shared_group,
    Mid_O,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od,
    Mid_O_LogExpSum,
    stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es,
    gqa_group_size,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    shared_batch_group_size = tl.load(b_mark_shared_group + cur_batch)
    if shared_batch_group_size == 0:
        return

    cur_batch_start = cur_batch - (shared_batch_group_size - 1)

    # ---- batch lane: 不再回卷索引，使用mask ----
    offs_b = tl.arange(0, BLOCK_BATCH)
    valid_b = offs_b < shared_batch_group_size
    batch_idx = cur_batch_start + offs_b

    # ---- head lane: 不再next_pow2回卷，使用mask ----
    offs_h = tl.arange(0, BLOCK_HEAD)
    valid_h = offs_h < gqa_group_size
    q_head_idx = cur_kv_head * gqa_group_size + offs_h

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # load seq len
    batch_seq_lens = tl.load(b_seq_len + batch_idx, mask=valid_b, other=0)
    max_seq_len = tl.max(batch_seq_lens, axis=0)

    kv_start = seq_start_block * BLOCK_SEQ
    if kv_start >= max_seq_len:
        return
    kv_end = tl.minimum(max_seq_len, kv_start + BLOCK_SEQ)

    # load Q (masked)
    off_q = (
        batch_idx[:, None, None] * stride_qb
        + q_head_idx[None, :, None] * stride_qh
        + offs_d[None, None, :] * stride_qd
    )
    q_mask = valid_b[:, None, None] & valid_h[None, :, None]
    q = tl.load(Q + off_q, mask=q_mask, other=0.0)

    # P1: 循环外提不变量
    q_flat = tl.reshape(q, (BLOCK_BATCH * BLOCK_HEAD, BLOCK_HEADDIM))

    sum_exp = tl.zeros([BLOCK_BATCH, BLOCK_HEAD], dtype=tl.float32)
    max_logic = tl.full([BLOCK_BATCH, BLOCK_HEAD], float("-inf"), dtype=tl.float32)
    acc = tl.zeros([BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM], dtype=tl.float32)

    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    offs_n_base = kv_start + tl.arange(0, BLOCK_N)

    # P1: 固定循环次数
    NUM_SUBBLOCK: tl.constexpr = BLOCK_SEQ // BLOCK_N
    for i in range(NUM_SUBBLOCK):
        offs_n_new = offs_n_base + i * BLOCK_N
        n_mask = offs_n_new < kv_end

        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new * stride_req_to_tokens_s,
            mask=n_mask,
            other=0,
        ).to(tl.int64)

        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
        off_v = k_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

        k = tl.load(K + off_k, mask=n_mask[:, None], other=0.0)
        v = tl.load(V + off_v, mask=n_mask[:, None], other=0.0)

        att_flat = tl.dot(q_flat, tl.trans(k))
        att = tl.reshape(att_flat, (BLOCK_BATCH, BLOCK_HEAD, BLOCK_N))
        att *= sm_scale

        visible_mask = offs_n_new[None, :] < batch_seq_lens[:, None]         # [B,N]
        combined_bn = valid_b[:, None] & n_mask[None, :] & visible_mask      # [B,N]
        combined = combined_bn[:, None, :] & valid_h[None, :, None]          # [B,H,N]
        att = tl.where(combined, att, float("-inf"))

        cur_max = tl.max(att, axis=2)
        new_max = tl.maximum(cur_max, max_logic)

        exp_logic = tl.exp(att - new_max[:, :, None])
        logic_scale = tl.exp(max_logic - new_max)

        acc *= logic_scale[:, :, None]
        exp_flat = tl.reshape(exp_logic, (BLOCK_BATCH * BLOCK_HEAD, BLOCK_N))
        acc_flat = tl.reshape(acc, (BLOCK_BATCH * BLOCK_HEAD, BLOCK_HEADDIM))
        acc_flat += tl.dot(exp_flat.to(q.dtype), v)
        acc = tl.reshape(acc_flat, (BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM))

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=2)
        max_logic = new_max

    # safe normalize
    safe_sum = tl.where(sum_exp > 0, sum_exp, 1.0)
    mid_o_val = acc / safe_sum[:, :, None]
    mid_lse_val = tl.where(sum_exp > 0, max_logic + tl.log(sum_exp), float("-inf"))

    off_mid_o = (
        batch_idx[:, None, None] * stride_mid_ob
        + q_head_idx[None, :, None] * stride_mid_oh
        + seq_start_block * stride_mid_os
        + offs_d[None, None, :] * stride_mid_od
    )
    off_mid_lse = (
        batch_idx[:, None] * stride_mid_o_eb
        + q_head_idx[None, :] * stride_mid_o_eh
        + seq_start_block * stride_mid_o_es
    )

    store_o_mask = valid_b[:, None, None] & valid_h[None, :, None]
    store_lse_mask = valid_b[:, None] & valid_h[None, :]
    tl.store(Mid_O + off_mid_o, mid_o_val, mask=store_o_mask)
    tl.store(Mid_O_LogExpSum + off_mid_lse, mid_lse_val, mask=store_lse_mask)


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
        run_config = {"BLOCK_N": 16, "num_warps": 2, "num_stages": 2}  # 默认配置

    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]
    BLOCK_BATCH = get_diverse_max_batch_shared_group_size()

    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert Req_to_tokens.is_cuda and B_req_idx.is_cuda and b_seq_len.is_cuda and b_mark_shared_group.is_cuda
    assert mid_out.is_cuda and mid_out_logsumexp.is_cuda
    BLOCK_SEQ = int(block_seq)
    assert BLOCK_SEQ % BLOCK_N == 0
    Lq, Lk = int(q.shape[2]), int(k.shape[2])
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    batch = int(B_req_idx.shape[0])
    kv_head_num = int(k.shape[1])
    q_head_num = int(q.shape[1])
    assert q_head_num % kv_head_num == 0
    gqa_group_size = q_head_num // kv_head_num
    BLOCK_HEAD = gqa_group_size  # P1: 不再 next_power_of_2 回卷
    
    sm_scale = 1.0 / (Lk ** 0.5)
    # 固定 grid（graph friendly）
    grid = (batch, kv_head_num, triton.cdiv(max_kv_len, BLOCK_SEQ))
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
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_SEQ=BLOCK_SEQ,
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
    gqa_group_size = 8
    q_head_dim = 128
    block_seq = 256
    out_dtype = torch.bfloat16

    batch_sizes = [1, 8, 16, 32, 64, 128]
    decode_lengths = [32, 64, 128, 256, 512, 1024, 2048, 8192, 16384]

    q_head_num = 32
    k_head_num = q_head_num // gqa_group_size

    Autotuner.start_autotune_warmup()
    # autotuing kernel
    for batch_size in batch_sizes:
        for length in decode_lengths:
            # Setup test tensors
            q = torch.randn(batch_size, q_head_num, q_head_dim, dtype=out_dtype, device="cuda")
            k = torch.randn(batch_size * length, k_head_num, q_head_dim, dtype=out_dtype, device="cuda")
            v = torch.randn(batch_size * length, k_head_num, q_head_dim, dtype=out_dtype, device="cuda")
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

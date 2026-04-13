import torch
import triton
import triton.language as tl
from typing import Optional

from lightllm.common.triton_utils.autotuner import autotune
from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size


def get_test_configs():
    configs = []
    for block_n in [16, 32, 64]:
        for num_warps in [2, 4, 8]:
            for num_stages in [2, 3, 4]:
                for num_ctas in [1, 2]:
                    configs.append(
                        {
                            "BLOCK_N": block_n,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "num_ctas": num_ctas,
                        }
                    )
    return configs


def get_static_key(q, k, max_kv_len):
    return {
        "gqa_group_size": int(q.shape[1] // k.shape[1]),
        "q_head_dim": int(q.shape[2]),
        "max_kv_len": int(max_kv_len),  # constexpr specialization
        "out_dtype": str(q.dtype),
    }


def get_run_key(q, max_kv_len):
    return int(q.shape[0]) * 1_000_000_000 + int(max_kv_len)


@triton.jit
def _fwd_kernel_mtp_diverse_single_token_fused(
    Q, stride_qb, stride_qh, stride_qd,
    K, stride_kbs, stride_kh, stride_kd,
    V, stride_vbs, stride_vh, stride_vd,
    sm_scale,
    Req_to_tokens, stride_req_to_tokens_b, stride_req_to_tokens_s,
    B_req_idx,
    b_seq_len,
    b_mark_shared_group,
    O, stride_ob, stride_oh, stride_od,
    gqa_group_size,
    max_kv_len,  # runtime int (for safety mask)
    BLOCK_HEAD: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_KV_LEN: tl.constexpr,  # compile-time loop upper bound
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)

    shared_batch_group_size = tl.load(b_mark_shared_group + cur_batch)
    if shared_batch_group_size == 0:
        return

    cur_batch_start = cur_batch - (shared_batch_group_size - 1)

    offs_b = tl.arange(0, BLOCK_BATCH)
    valid_b = offs_b < shared_batch_group_size
    batch_idx = cur_batch_start + offs_b

    offs_h = tl.arange(0, BLOCK_HEAD)
    valid_h = offs_h < gqa_group_size
    q_head_idx = cur_kv_head * gqa_group_size + offs_h

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    batch_seq_lens = tl.load(b_seq_len + batch_idx, mask=valid_b, other=0)
    max_seq_len = tl.max(batch_seq_lens, axis=0)
    if max_seq_len == 0:
        return

    off_q = (
        batch_idx[:, None, None] * stride_qb
        + q_head_idx[None, :, None] * stride_qh
        + offs_d[None, None, :] * stride_qd
    )
    q_mask = valid_b[:, None, None] & valid_h[None, :, None]
    q = tl.load(Q + off_q, mask=q_mask, other=0.0)
    q_flat = tl.reshape(q, (BLOCK_BATCH * BLOCK_HEAD, BLOCK_HEADDIM))

    sum_exp = tl.zeros([BLOCK_BATCH, BLOCK_HEAD], dtype=tl.float32)
    max_logic = tl.full([BLOCK_BATCH, BLOCK_HEAD], float("-inf"), dtype=tl.float32)
    acc = tl.zeros([BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM], dtype=tl.float32)

    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    # 全序列直接做 online softmax 聚合（融合 stage2）
    for kv_start in tl.static_range(0, MAX_KV_LEN, BLOCK_N):
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        n_mask = (offs_n < max_seq_len) & (offs_n < max_kv_len)

        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n * stride_req_to_tokens_s,
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

        visible_mask = offs_n[None, :] < batch_seq_lens[:, None]           # [B, N]
        combined_bn = valid_b[:, None] & n_mask[None, :] & visible_mask
        combined = combined_bn[:, None, :] & valid_h[None, :, None]        # [B, H, N]
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

    safe_sum = tl.where(sum_exp > 0, sum_exp, 1.0)
    out_val = acc / safe_sum[:, :, None]

    off_o = (
        batch_idx[:, None, None] * stride_ob
        + q_head_idx[None, :, None] * stride_oh
        + offs_d[None, None, :] * stride_od
    )
    store_mask = valid_b[:, None, None] & valid_h[None, :, None]
    tl.store(O + off_o, out_val, mask=store_mask)


@autotune(
    kernel_name="_fwd_kernel_mtp_diverse_single_token_fused:v1",
    configs_gen_func=get_test_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    mutates_args=["O"],
)
@torch.no_grad()
def mtp_diverse_single_token_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_mark_shared_group: torch.Tensor,
    max_kv_len: int,
    O: torch.Tensor,
    run_config: Optional[dict] = None,
):
    if not run_config:
        run_config = {"BLOCK_N": 16, "num_warps": 4, "num_stages": 2, "num_ctas": 1}

    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]
    num_ctas = run_config.get("num_ctas", 1)

    Lq, Lk = int(q.shape[2]), int(k.shape[2])
    assert Lq == Lk and Lk in {16, 32, 64, 128}
    batch = int(q.shape[0])
    kv_head_num = int(k.shape[1])
    q_head_num = int(q.shape[1])
    assert q_head_num % kv_head_num == 0

    gqa_group_size = q_head_num // kv_head_num
    BLOCK_HEAD = gqa_group_size
    BLOCK_BATCH = get_diverse_max_batch_shared_group_size()
    sm_scale = 1.0 / (Lk ** 0.5)

    grid = (batch, kv_head_num)
    _fwd_kernel_mtp_diverse_single_token_fused[grid](
        Q=q,
        stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qd=q.stride(2),
        K=k,
        stride_kbs=k.stride(0), stride_kh=k.stride(1), stride_kd=k.stride(2),
        V=v,
        stride_vbs=v.stride(0), stride_vh=v.stride(1), stride_vd=v.stride(2),
        sm_scale=sm_scale,
        Req_to_tokens=Req_to_tokens,
        stride_req_to_tokens_b=Req_to_tokens.stride(0),
        stride_req_to_tokens_s=Req_to_tokens.stride(1),
        B_req_idx=B_req_idx,
        b_seq_len=b_seq_len,
        b_mark_shared_group=b_mark_shared_group,
        O=O,
        stride_ob=O.stride(0), stride_oh=O.stride(1), stride_od=O.stride(2),
        gqa_group_size=gqa_group_size,
        max_kv_len=max_kv_len,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_BATCH=BLOCK_BATCH,
        BLOCK_HEADDIM=Lk,
        BLOCK_N=BLOCK_N,
        MAX_KV_LEN=max_kv_len,
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=num_ctas,
    )
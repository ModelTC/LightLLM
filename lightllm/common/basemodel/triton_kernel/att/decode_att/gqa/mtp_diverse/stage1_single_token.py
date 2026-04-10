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
from lightllm.utils.device_utils import is_hopper


def get_test_configs():
    configs = []
    for block_n in [16, 32, 64]:
        for num_warps in [2, 4, 8]:
            for num_stages in [2, 3, 4]:
                for warp_specialize in [True, False] if is_hopper() else [False]:
                    # warp_specialize only support hopper
                    configs.append(
                        {
                            "BLOCK_N": block_n,
                            "num_warps": num_warps,
                            "num_stages": num_stages,
                            "warp_specialize": warp_specialize,
                        }
                    )
    return configs


def get_static_key(q, k, block_batch):
    key_params = {
        "gqa_group_size": int(q.shape[1] // k.shape[1]),
        "q_head_dim": int(q.shape[2]),
        "block_batch": block_batch,
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
    Mid_O,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    Mid_O_LogExpSum,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    block_index = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_batch = tl.program_id(2)
    grid_block_num = tl.num_programs(0)

    shared_batch_group_size = tl.load(b_mark_shared_group + cur_batch)
    if shared_batch_group_size == 0:
        return

    cur_batch_start = cur_batch - (shared_batch_group_size - 1)

    # ---- batch lane: 不再回卷索引，使用mask ----
    offs_b = tl.arange(0, BLOCK_BATCH)
    batch_idx = tl.where(offs_b < shared_batch_group_size, cur_batch_start + offs_b, cur_batch)

    # load seq len
    batch_seq_lens = tl.load(b_seq_len + batch_idx)
    max_seq_len = tl.max(batch_seq_lens, axis=0)
    block_num = tl.cdiv(max_seq_len, BLOCK_N)
    if block_index >= block_num:
        return

    batch_seq_lens = tl.broadcast_to(batch_seq_lens[:, None], (BLOCK_BATCH, BLOCK_HEAD))
    batch_seq_lens = batch_seq_lens.reshape((BLOCK_BATCH * BLOCK_HEAD,))

    # ---- head lane: 不再next_pow2回卷，使用mask  ---    -
    offs_h = tl.arange(0, BLOCK_HEAD)
    q_head_idx = tl.where(offs_h < gqa_group_size, cur_kv_head * gqa_group_size + offs_h, cur_kv_head * gqa_group_size)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    off_q = batch_idx[:, None, None] * stride_qb + q_head_idx[None, :, None] * stride_qh + offs_d[None, None, :]
    q = tl.load(Q + off_q)
    q_flat = tl.reshape(q, (BLOCK_BATCH * BLOCK_HEAD, BLOCK_HEADDIM))

    sum_exp = tl.zeros([BLOCK_BATCH * BLOCK_HEAD], dtype=tl.float32)
    max_logic = tl.full([BLOCK_BATCH * BLOCK_HEAD], float("-inf"), dtype=tl.float32)
    acc = tl.zeros([BLOCK_BATCH * BLOCK_HEAD, BLOCK_HEADDIM], dtype=tl.float32)

    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    for iter_block_index in tl.range(block_index, block_num, grid_block_num, warp_specialize=warp_specialize):
        offs_n_new = iter_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_n_refator = tl.where(offs_n_new < max_seq_len, offs_n_new, max_seq_len - 1)

        k_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_refator * stride_req_to_tokens_s,
        ).to(tl.int64)
        off_k = k_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None]
        off_v = k_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :]
        k = tl.load(K + off_k)
        v = tl.load(V + off_v)
        att = tl.dot(q_flat, k)
        att *= sm_scale
        att = tl.where(offs_n_new[None, :] < batch_seq_lens[:, None], att, -1000000000.0)
        cur_max = tl.max(att, axis=1)
        new_max = tl.maximum(cur_max, max_logic)

        exp_logic = tl.exp(att - new_max[:, None])
        logic_scale = tl.exp(max_logic - new_max)

        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(v.dtype), v)

        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max

    mid_o_val = acc / sum_exp[:, None]
    mid_lse_val = max_logic + tl.log(sum_exp)

    off_mid_o = (
        batch_idx[:, None, None] * stride_mid_ob
        + q_head_idx[None, :, None] * stride_mid_oh
        + block_index * stride_mid_os
        + offs_d[None, None, :] * stride_mid_od
    )
    off_mid_lse = (
        batch_idx[:, None] * stride_mid_o_eb + q_head_idx[None, :] * stride_mid_o_eh + block_index * stride_mid_o_es
    )

    tl.store(Mid_O + off_mid_o, mid_o_val.reshape((BLOCK_BATCH, BLOCK_HEAD, BLOCK_HEADDIM)))
    tl.store(Mid_O_LogExpSum + off_mid_lse, mid_lse_val.reshape((BLOCK_BATCH, BLOCK_HEAD)))


@autotune(
    kernel_name="_fwd_kernel_mtp_diverse_stage1_single_token:v2",
    configs_gen_func=get_test_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    mutates_args=["mid_out", "mid_out_logsumexp"],
)
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
    block_batch: int,
    run_config: Optional[dict] = None,
):
    """
    MTP Diverse Attention Stage1 - Single Token Per Request Mode

    b_seq_len: 每个请求可见的 KV 数量，组内递增
    例如组内 [q1, q2, q3, q4] 对应 b_seq_len [2, 3, 4, 5]
    """
    if not run_config:
        run_config = {"BLOCK_N": 16, "num_warps": 2, "num_stages": 2, "warp_specialize": False}  # 默认配置

    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]
    warp_specialize = run_config.get("warp_specialize", False)
    BLOCK_BATCH = triton.next_power_of_2(block_batch)

    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert Req_to_tokens.is_cuda and B_req_idx.is_cuda and b_seq_len.is_cuda and b_mark_shared_group.is_cuda
    assert mid_out.is_cuda and mid_out_logsumexp.is_cuda

    Lq, Lk = int(q.shape[2]), int(k.shape[2])
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128}
    batch = int(B_req_idx.shape[0])
    kv_head_num = int(k.shape[1])
    q_head_num = int(q.shape[1])
    assert q_head_num % kv_head_num == 0
    gqa_group_size = q_head_num // kv_head_num
    BLOCK_HEAD = triton.next_power_of_2(gqa_group_size)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1

    grid_num = mid_out.shape[2]

    sm_scale = 1.0 / (Lk ** 0.5)
    # 固定 grid（graph friendly）
    grid = (grid_num, kv_head_num, batch)
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
        BLOCK_HEADDIM=Lk,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
        warp_specialize=warp_specialize,
    )
    return BLOCK_N


if __name__ == "__main__":
    from lightllm.utils.envs_utils import get_triton_autotune_level

    if get_triton_autotune_level() != 2:
        raise Exception("you need set env LIGHTLLM_TRITON_AUTOTUNE_LEVEL=2 to start program.")

    # static params
    gqa_group_size = 4
    q_head_dim = 128
    block_batch = 4
    out_dtype = torch.bfloat16

    batch_sizes = [1, 8, 16, 32, 64, 128]
    decode_lengths = [32, 64, 128, 256, 512, 1024, 2048]

    q_head_num = gqa_group_size
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
            b_mark_shared_group = torch.ones(batch_size, dtype=torch.int32, device="cuda")

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
                block_batch=block_batch,
            )

    Autotuner.end_autotune_warmup()

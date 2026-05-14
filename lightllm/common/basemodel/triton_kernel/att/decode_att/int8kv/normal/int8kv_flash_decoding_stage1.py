import torch
import triton
import triton.language as tl
from typing import Optional
from lightllm.common.triton_utils.autotuner import autotune, Autotuner


@triton.jit
def _fwd_kernel_flash_decode_normal_stage1(
    Q,
    stride_qbs,
    stride_qh,
    stride_qd,
    K,
    K_scale,
    stride_kbs,
    stride_kh,
    stride_kd,
    V,
    V_scale,
    stride_vbs,
    stride_vh,
    stride_vd,
    sm_scale,
    Req_to_tokens,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    B_req_idx,
    b_seq_len,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
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
    KV_QUANT_GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    block_index = tl.program_id(2)
    grid_block_num = tl.num_programs(2)
    cur_batch_seq_len = tl.load(b_seq_len + cur_batch)
    req_total_block_num = tl.cdiv(cur_batch_seq_len, BLOCK_SEQ)
    if block_index >= req_total_block_num:
        return

    cur_q_head_range = cur_kv_head * gqa_group_size + tl.arange(0, BLOCK_HEAD)
    q_head_end_index = (cur_kv_head + 1) * gqa_group_size
    cur_q_head_range = tl.where(cur_q_head_range < q_head_end_index, cur_q_head_range, cur_kv_head * gqa_group_size)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_d_scale = tl.arange(0, NUM_GROUPS)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    off_q = cur_batch * stride_qbs + cur_q_head_range[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(Q + off_q)  # (BLOCK_HEAD, BLOCK_HEADDIM)

    sum_exp = tl.zeros([BLOCK_HEAD], dtype=tl.float32)
    max_logic = tl.zeros([BLOCK_HEAD], dtype=tl.float32) - float("inf")
    acc = tl.zeros([BLOCK_HEAD, BLOCK_HEADDIM], dtype=tl.float32)

    for iter_block_index in range(block_index, req_total_block_num, grid_block_num):
        cur_batch_start_index = iter_block_index * BLOCK_SEQ
        cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

        offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
        block_n_size = tl.cdiv(cur_batch_end_index - cur_batch_start_index, BLOCK_N)

        for start_n in range(0, block_n_size, 1):
            offs_n_new = start_n * BLOCK_N + offs_n
            n_mask = offs_n_new < cur_batch_end_index
            k_loc = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
                mask=n_mask,
                other=0,
            ).to(tl.int64)
            off_k_base = k_loc * stride_kbs + cur_kv_head * stride_kh
            off_k = off_k_base[None, :] + offs_d[:, None]
            # off_k_base // KV_QUANT_GROUP_SIZE 是一种取巧计算stride的方式
            off_k_scale = off_k_base[None, :] // KV_QUANT_GROUP_SIZE + offs_d_scale[:, None]
            k = tl.load(K + off_k, mask=n_mask[None, :], other=0)
            k = tl.reshape(k, (NUM_GROUPS, KV_QUANT_GROUP_SIZE, BLOCK_N))
            k_scale = tl.load(K_scale + off_k_scale, mask=n_mask[None, :], other=0.0)
            k_scale = tl.reshape(k_scale, (NUM_GROUPS, 1, BLOCK_N))
            k = k * k_scale
            k = tl.reshape(k, (BLOCK_HEADDIM, BLOCK_N))
            att_value = tl.dot(q, k.to(q.dtype))
            att_value *= sm_scale
            att_value = tl.where(n_mask[None, :], att_value, float("-inf"))
            off_v = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :]
            v = tl.load(
                V + off_v,
                mask=n_mask[:, None],
                other=0,
            )
            v = tl.reshape(v, (BLOCK_N, NUM_GROUPS, KV_QUANT_GROUP_SIZE))
            v_scale = tl.load(
                V_scale + off_k_scale,
                mask=n_mask[None, :],
                other=0.0,
            )
            v_scale = tl.trans(v_scale)
            v_scale = tl.reshape(v_scale, (BLOCK_N, NUM_GROUPS, 1))
            v = v * v_scale
            v = tl.reshape(v, (BLOCK_N, BLOCK_HEADDIM))

            cur_max_logic = tl.max(att_value, axis=1)
            new_max_logic = tl.maximum(cur_max_logic, max_logic)

            exp_logic = tl.exp(att_value - new_max_logic[:, None])
            logic_scale = tl.exp(max_logic - new_max_logic)
            acc *= logic_scale[:, None]
            acc += tl.dot(exp_logic.to(q.dtype), v.to(q.dtype))

            sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
            max_logic = new_max_logic

    off_mid_o = (
        cur_batch * stride_mid_ob
        + cur_q_head_range[:, None] * stride_mid_oh
        + block_index * stride_mid_os
        + offs_d[None, :]
    )
    off_mid_o_logexpsum = cur_batch * stride_mid_o_eb + cur_q_head_range * stride_mid_o_eh + block_index
    tl.store(
        Mid_O + off_mid_o,
        (acc / sum_exp[:, None]).reshape(BLOCK_HEAD, BLOCK_HEADDIM),
    )
    tl.store(
        Mid_O_LogExpSum + off_mid_o_logexpsum,
        (max_logic + tl.log(sum_exp)),
    )
    return


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


def get_static_key(q, k, k_scale, block_seq):
    key_params = {
        "kv_quant_group_size": k.shape[-1] // k_scale.shape[-1],
        "gqa_group_size": int(q.shape[1] // k.shape[1]),
        "q_head_dim": int(q.shape[2]),
        "block_seq": block_seq,
        "out_dtype": str(q.dtype),
    }
    return key_params


def get_run_key(q, max_len_in_batch):
    batch_size = q.shape[0]
    return batch_size * 1000 * 1000 * 1000 + max_len_in_batch


@autotune(
    kernel_name="_fwd_kernel_flash_decode_normal_stage1:v3",
    configs_gen_func=get_test_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    mutates_args=["mid_out", "mid_out_logsumexp"],
)
def flash_decode_stage1(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    v: torch.Tensor,
    v_scale: torch.Tensor,
    Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor,
    B_seq_len: torch.Tensor,
    max_len_in_batch: int,
    mid_out: torch.Tensor,
    mid_out_logsumexp: torch.Tensor,
    block_seq: int,
    run_config: Optional[dict] = None,
):
    """ """
    if not run_config:

        run_config = {
            "BLOCK_N": 16,
            "num_warps": 2,
            "num_stages": 2,
        }

    BLOCK_N = run_config["BLOCK_N"]
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]

    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    BLOCK_SEQ = block_seq
    assert BLOCK_SEQ % BLOCK_N == 0
    # shape constraints
    Lq, Lk = q.shape[-1], k.shape[-1]
    assert Lq == Lk
    assert Lk in {16, 32, 64, 128, 256}
    if Lk == 256:
        BLOCK_N = min(BLOCK_N, 16)
    assert BLOCK_SEQ % BLOCK_N == 0
    sm_scale = 1.0 / (Lk ** 0.5)
    batch, kv_head_num = B_req_idx.shape[0], k.shape[1]
    gqa_group_size = q.shape[1] // k.shape[1]
    assert triton.next_power_of_2(Lk) == Lk
    KV_QUANT_GROUP_SIZE = v.shape[-1] // v_scale.shape[-1]
    assert triton.next_power_of_2(KV_QUANT_GROUP_SIZE) == KV_QUANT_GROUP_SIZE
    BLOCK_HEAD = triton.next_power_of_2(gqa_group_size)

    assert k.stride() == v.stride()
    NUM_GROUPS = Lk // KV_QUANT_GROUP_SIZE
    assert triton.next_power_of_2(NUM_GROUPS) == NUM_GROUPS

    assert k.stride() == v.stride()
    block_num = mid_out.shape[2]
    grid = (batch, kv_head_num, block_num)
    _fwd_kernel_flash_decode_normal_stage1[grid](
        Q=q,
        stride_qbs=q.stride(0),
        stride_qh=q.stride(1),
        stride_qd=q.stride(2),
        K=k,
        K_scale=k_scale,
        stride_kbs=k.stride(0),
        stride_kh=k.stride(1),
        stride_kd=k.stride(2),
        V=v,
        V_scale=v_scale,
        stride_vbs=v.stride(0),
        stride_vh=v.stride(1),
        stride_vd=v.stride(2),
        sm_scale=sm_scale,
        Req_to_tokens=Req_to_tokens,
        stride_req_to_tokens_b=Req_to_tokens.stride(0),
        stride_req_to_tokens_s=Req_to_tokens.stride(1),
        B_req_idx=B_req_idx,
        b_seq_len=B_seq_len,
        Mid_O=mid_out,
        stride_mid_ob=mid_out.stride(0),
        stride_mid_oh=mid_out.stride(1),
        stride_mid_os=mid_out.stride(2),
        stride_mid_od=mid_out.stride(3),
        Mid_O_LogExpSum=mid_out_logsumexp,  # [batch, head, seq_block_num]
        stride_mid_o_eb=mid_out_logsumexp.stride(0),
        stride_mid_o_eh=mid_out_logsumexp.stride(1),
        stride_mid_o_es=mid_out_logsumexp.stride(2),
        gqa_group_size=gqa_group_size,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=block_seq,
        BLOCK_HEADDIM=Lk,
        BLOCK_N=BLOCK_N,
        KV_QUANT_GROUP_SIZE=KV_QUANT_GROUP_SIZE,
        NUM_GROUPS=NUM_GROUPS,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return


if __name__ == "__main__":
    from lightllm.utils.envs_utils import get_triton_autotune_level

    if get_triton_autotune_level() != 2:
        raise Exception("you need set env LIGHTLLM_TRITON_AUTOTUNE_LEVEL=2 to start program.")

    # static params
    kv_quant_group_size = 8
    gqa_group_size = 4
    q_head_dim = 128
    block_seq = 256
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
            k = torch.ones(batch_size * length, 1, q_head_dim, dtype=torch.int8, device="cuda")
            k_scale = torch.randn(
                batch_size * length, 1, q_head_dim // kv_quant_group_size, dtype=torch.float32, device="cuda"
            )
            v = torch.ones(batch_size * length, 1, q_head_dim, dtype=torch.int8, device="cuda")
            v_scale = torch.randn(
                batch_size * length, 1, q_head_dim // kv_quant_group_size, dtype=torch.float32, device="cuda"
            )
            Req_to_tokens = torch.arange(0, batch_size * length, dtype=torch.int32, device="cuda").view(
                batch_size, length
            )
            B_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
            B_seq_len = torch.full((batch_size,), length, dtype=torch.int32, device="cuda")

            if batch_size <= 16:
                block_num = 128
            elif batch_size <= 64:
                block_num = 64
            else:
                block_num = 32

            mid_out = torch.zeros(batch_size, q_head_num, block_num, q_head_dim, dtype=out_dtype, device="cuda")
            mid_out_logsumexp = torch.zeros(batch_size, q_head_num, block_num, dtype=out_dtype, device="cuda")

            flash_decode_stage1(
                q=q,
                k=k,
                k_scale=k_scale,
                v=v,
                v_scale=v_scale,
                Req_to_tokens=Req_to_tokens,
                B_req_idx=B_req_idx,
                B_seq_len=B_seq_len,
                max_len_in_batch=length,
                mid_out=mid_out,
                mid_out_logsumexp=mid_out_logsumexp,
                block_seq=block_seq,
            )

    Autotuner.end_autotune_warmup()

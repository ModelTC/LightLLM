import torch
import triton
import triton.language as tl
import itertools
from lightllm.utils.device_utils import is_tesla
from lightllm.common.triton_utils.autotuner import autotune


@triton.jit
def _glm_fwd_kernel_with_v(
    Q_nope,
    Q_rope,
    K_nope,
    K_rope,
    V,
    sm_scale,
    B_Start_Loc,
    B_Kv_Start_Loc,
    B_Seqlen,
    Out,
    stride_q_bs,
    stride_q_h,
    stride_q_d,
    stride_q_rope_bs,
    stride_q_rope_h,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_k_rope_bs,
    stride_k_rope_h,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    b_prompt_cache_len,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_NOPE_DIM: tl.constexpr,
    BLOCK_ROPE_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_k_head = cur_head

    cur_batch_in_q_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_in_kv_start_index = tl.load(B_Kv_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_rope_d = tl.arange(0, BLOCK_ROPE_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    nope_valid_mask = offs_d < ACTUAL_NOPE_DIM

    off_q = (
        (cur_batch_in_q_start_index + offs_m[:, None]) * stride_q_bs
        + cur_head * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    off_q_rope = (
        (cur_batch_in_q_start_index + offs_m[:, None]) * stride_q_rope_bs
        + cur_head * stride_q_rope_h
        + offs_rope_d[None, :]
    )

    off_k = offs_n[None, :] * stride_k_bs + cur_k_head * stride_k_h + offs_d[:, None] * stride_k_d
    off_k_rope = offs_n[None, :] * stride_k_rope_bs + offs_rope_d[:, None]

    off_v = offs_n[:, None] * stride_vbs + cur_k_head * stride_vh + offs_d[None, :]

    seq_mask_q = offs_m[:, None] < cur_batch_seq_len
    q = tl.load(Q_nope + off_q, mask=seq_mask_q & nope_valid_mask[None, :], other=0.0)
    q_rope = tl.load(Q_rope + off_q_rope, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K_nope + off_k
    k_rope_ptrs = K_rope + off_k_rope
    v_ptrs = V + off_v

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum((start_m + 1) * BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)

    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        seq_mask_k = (start_n + offs_n[None, :]) < block_end_loc

        k = tl.load(
            k_ptrs + (cur_batch_in_kv_start_index + start_n) * stride_k_bs,
            mask=seq_mask_k & nope_valid_mask[:, None],
            other=0.0,
        )
        k_rope = tl.load(
            k_rope_ptrs + (cur_batch_in_kv_start_index + start_n) * stride_k_rope_bs,
            mask=(start_n + offs_n[None, :]) < block_end_loc,
            other=0.0,
        )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk += tl.dot(q_rope, k_rope)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :], qk, float("-100000000.0"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptrs + (cur_batch_in_kv_start_index + start_n) * stride_vbs,
            mask=(start_n + offs_n[:, None]) < block_end_loc,
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (cur_batch_in_q_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :]
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


def get_autotune_configs():
    configs = []
    block_sizes = [32, 64, 128] if not is_tesla() else [16, 32, 64]
    num_warps_options = [4, 8]
    num_stages_options = [1, 2]

    for block_size, num_warps, num_stages in itertools.product(block_sizes, num_warps_options, num_stages_options):
        configs.append(
            {
                "BLOCK": block_size,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )
    return configs


def get_static_key(q_nope, q_rope, v):
    return {
        "q_nope_dim": q_nope.shape[-1],
        "q_rope_dim": q_rope.shape[-1],
        "v_dim": v.shape[-1],
        "num_heads": q_nope.shape[1],
        "dtype": str(q_nope.dtype),
    }


def get_run_key(max_input_len):
    return max_input_len


@autotune(
    kernel_name="glm_context_attention_fwd_with_v:v1",
    configs_gen_func=get_autotune_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    mutates_args=["o"],
)
@torch.no_grad()
def glm_context_attention_fwd_with_v(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    v,
    o,
    b_start_loc,
    b_kv_start_loc,
    b_seq_len,
    b_prompt_cache_len,
    max_input_len,
    softmax_scale,
    run_config=None,
):
    ACTUAL_NOPE_DIM = q_nope.shape[-1]
    BLOCK_DMODEL = v.shape[-1]
    BLOCK_ROPE_DMODEL = q_rope.shape[-1]

    if run_config is None:
        BLOCK = 64 if not is_tesla() else 32
        num_warps = 4
        num_stages = 1
    else:
        BLOCK = run_config["BLOCK"]
        num_warps = run_config["num_warps"]
        num_stages = run_config["num_stages"]

    if q_nope.dtype == torch.float32:
        BLOCK = BLOCK // 4

    sm_scale = softmax_scale * 1.4426950408889634
    batch, head = b_seq_len.shape[0], q_nope.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))

    _glm_fwd_kernel_with_v[grid](
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        v,
        sm_scale,
        b_start_loc,
        b_kv_start_loc,
        b_seq_len,
        o,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_nope.stride(2),
        k_rope.stride(0),
        k_rope.stride(1),
        v.stride(0),
        v.stride(1),
        o.stride(0),
        o.stride(1),
        b_prompt_cache_len=b_prompt_cache_len,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_NOPE_DIM=ACTUAL_NOPE_DIM,
        BLOCK_ROPE_DMODEL=BLOCK_ROPE_DMODEL,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

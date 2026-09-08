# SPDX-License-Identifier: Apache-2.0
#                            MIT
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# Extracted from fused_recurrent.py — directly launches the triton kernel
# without a torch.autograd.Function wrapper. Used by the MTP decode
# verify path of the GDN (Gated DeltaNet) layer in Qwen3Next.
#
# Upstream source: flash-linear-attention / fused-recurrent gated delta rule.
# https://github.com/fla-org/flash-linear-attention
# ruff: noqa: E501

import torch
import triton
import triton.language as tl
from typing import Optional
from lightllm.common.triton_utils.autotuner import autotune, AutotuneKernelType


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def _fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    ssm_state_write_indices,
    num_accepted_tokens,
    A_log,
    dt_bias,
    a_raw,
    b_raw,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    stride_q_tok: tl.constexpr,  # token stride in q/k/v/a/b
    stride_k_tok: tl.constexpr,
    stride_v_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_o_tok: tl.constexpr,  # token stride in output ([HV, V] contiguous → HV*V)
    stride_init_state_token: tl.constexpr,  # stride per slot in initial/final state
    stride_final_state_token: tl.constexpr,
    stride_state_hv: tl.constexpr,  # stride per HV-head inside a state slot (K*V)
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    stride_write_indices_seq: tl.constexpr,
    stride_write_indices_tok: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)
    bos, eos = (
        tl.load(cu_seqlens + i_n).to(tl.int64),
        tl.load(cu_seqlens + i_n + 1).to(tl.int64),
    )
    T = eos - bos

    if T == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q_tok + i_h * K + o_k
    p_k = k + bos * stride_k_tok + i_h * K + o_k
    p_v = v + bos * stride_v_tok + i_hv * V + o_v
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    p_a_raw = a_raw + bos * stride_a_tok + i_hv
    p_b_raw = b_raw + bos * stride_b_tok + i_hv

    p_o = o + bos * stride_o_tok + i_hv * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    p_h0 = h0 + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64) * stride_init_state_token
    p_h0 = p_h0 + i_hv * stride_state_hv + o_k[:, None] * V + o_v[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for i_t in tl.range(0, T, num_stages=NUM_STAGES):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_a = tl.load(p_a_raw).to(tl.float32)
        x = b_a + b_dt_bias
        softplus_x = tl.where(
            SOFTPLUS_BETA * x <= SOFTPLUS_THRESHOLD,
            (1.0 / SOFTPLUS_BETA) * tl.log(1.0 + tl.exp(SOFTPLUS_BETA * x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x
        b_h *= tl.exp(b_g)
        b_b = tl.load(p_b_raw).to(tl.float32)
        b_beta = tl.sigmoid(b_b)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        write_idx = tl.load(ssm_state_write_indices + i_n * stride_write_indices_seq + i_t).to(tl.int64)
        p_ht = ht + write_idx * stride_final_state_token
        p_ht = p_ht + i_hv * stride_state_hv + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += stride_q_tok
        p_k += stride_k_tok
        p_o += stride_o_tok
        p_v += stride_v_tok
        p_a_raw += stride_a_tok
        p_b_raw += stride_b_tok


# ---------------------------------------------------------------------------
# Public API — directly launches the triton kernel (no autograd.Function)
# ---------------------------------------------------------------------------


def get_test_configs():
    # BK 必须覆盖整个 K 维归约，只搜索 V 维分块、warp 数和循环流水级数；保留默认 BV=8/warps=1/stages=1。
    return [
        {"BV": bv, "num_warps": num_warps, "num_stages": num_stages}
        for bv in [8, 16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ]


def get_static_key(q, v, initial_state, ssm_state_write_indices):
    return {
        "head_k_dim": q.shape[-1],
        "head_v_dim": v.shape[-1],
        "num_v_heads": v.shape[2],
        "gva_group_size": v.shape[2] // q.shape[2],
        "mtp_size": ssm_state_write_indices.shape[1],
        "q_dtype": str(q.dtype),
        "state_dtype": str(initial_state.dtype),
    }


def get_run_key(q, ssm_state_write_indices):
    # 线性 attention 的历史信息已压缩到固定大小的 SSM 状态，计算量不随历史 KV 长度增长。
    # head 数由 static key 区分，run key 只按本轮 Q token 数分桶，同桶内不同 MTP 分组共用配置。
    # 只读取形状，不读取 GPU 内容，保证正常查找可用于 Graph 捕获。
    num_q_tokens = q.shape[1]
    # 按 4 个 MTP 请求的 token 宽度向上分桶，使相近长度复用配置。
    mtp_size = ssm_state_write_indices.shape[1]
    token_bucket_size = 4 * mtp_size
    return triton.cdiv(num_q_tokens, token_bucket_size) * token_bucket_size


def rebuild_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    ssm_state_write_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    **kwargs,
):
    # Graph 初始化使用 HOLD 请求：固定 MTP 可能重复写同一状态槽，动态 MTP 则全部是空序列。
    # 直接计时会产生状态竞争或只测到提前返回，因此搜索前重建合法的 MTP 分组和互不重叠的状态索引。
    # 保留 Q/K/V、gate 的形状和 stride，以及 cu_seqlens 的长度，使 benchmark 与 run key 一致。
    num_tokens = q.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    mtp_size = ssm_state_write_indices.shape[1]
    assert num_tokens > 0 and mtp_size > 0, "MTP autotuning requires non-empty tokens and state index rows"
    active_seqs = triton.cdiv(num_tokens, mtp_size)
    assert active_seqs <= num_seqs, "Not enough sequence rows for the MTP tokens"
    num_state_slots = initial_state.shape[0]
    # 有效 token 必须拥有互不重叠的状态槽，避免取余后不同请求并发读写同一槽位。
    assert num_tokens <= num_state_slots, "Not enough SSM state slots for autotuning"

    # 按完整 MTP 宽度分组，末组可更短，动态布局多余的行仍为空序列。
    # 例如 7 个 token、MTP 宽度 3、7 行序列，累积长度为 [0, 3, 6, 7, 7, 7, 7, 7]。
    cu_seqlens = torch.tensor(
        [min(row * mtp_size, num_tokens) for row in range(num_seqs + 1)],
        dtype=cu_seqlens.dtype,
        device=cu_seqlens.device,
    )
    # 按状态池总槽数取余，使读写索引都落在 [0, num_state_slots) 内。
    # 容量检查保证有效 token 的槽号不变且互不重叠；仅空序列及末组未使用的索引可能回绕。
    state_indices = [
        [(row * mtp_size + offset) % num_state_slots for offset in range(mtp_size)] for row in range(num_seqs)
    ]
    ssm_state_indices = torch.tensor(state_indices, dtype=ssm_state_indices.dtype, device=ssm_state_indices.device)
    ssm_state_write_indices = ssm_state_indices.to(dtype=ssm_state_write_indices.dtype)
    num_accepted_tokens = torch.ones_like(num_accepted_tokens)

    # 直接复用 initial_state，不新建或 clone 状态池，避免服务显存紧张时因额外分配触发 OOM。
    # benchmark 会原地覆盖池内前 num_tokens 个槽且不恢复；只能在启动阶段使用，
    # 此时这些槽中不能有需要保留的请求状态。这里只重建分组和索引，不初始化状态数值。
    return (
        q,
        k,
        v,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
        ssm_state_write_indices,
        num_accepted_tokens,
        A_log,
        dt_bias,
        a_raw,
        b_raw,
    ), kwargs


@autotune(
    kernel_name="_mtp_fused_recurrent_gated_delta_rule_fwd_kernel:v1",
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    configs_gen_func=get_test_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    rebuild_input_func=rebuild_inputs,
    # 历史配置预热直接使用调用方的状态池，每额外执行一次都会推进 SSM 状态，影响正式计算。
    # 状态池很大，不适合通过 clone 保存/恢复，因此在所有阶段都关闭这类额外预热。
    warmup_all_exist_config=False,
    # 不标记 initial_state 为可变参数：服务状态池很大，autotuner 在预热/调优时反复 clone
    # 会额外占用大量显存，可能频繁触发 OOM，因此调优直接复用并原地更新传入的状态池。
    # 循环次数、访存地址和计算路径不依赖状态数值，理论上不影响调优对配置性能的选优。
    # 实际调优搜索仍会写入状态池，仅用于启动阶段尚无有效请求状态时。
    # mutates_args=["initial_state"],
)
@torch.no_grad()
def mtp_fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    ssm_state_write_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    run_config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused recurrent gated delta rule with fused gating (GDN layer).

    Directly launches the triton kernel — no ``torch.autograd.Function``.

    Args:
        q:  ``[1, T, H, K]`` queries.
        k:  ``[1, T, H, K]`` keys.
        v:  ``[1, T, HV, V]`` values (GVA when HV > H).
        initial_state: ``[num_slots, HV, K, V]`` SSM state pool, updated in-place.
        cu_seqlens: ``[N+1]`` int64 cumulative sequence lengths for the
            varlen (MTP verify) path.
        ssm_state_indices: ``[N, S+1]`` int32 slot indices (2D).
        ssm_state_write_indices: ``[N, S+1]`` int32 write slot indices.
        num_accepted_tokens: ``[N]`` int32.  The read offset for each
            sequence is ``num_accepted_tokens[i] - 1``.
        A_log: ``[HV]`` per-head log decay.
        dt_bias: ``[HV]`` per-head dt bias.
        a_raw: ``[T, HV]`` raw alpha.
        b_raw: ``[T, HV]`` raw beta.
        run_config: Optional ``BV``, ``num_warps`` and ``num_stages`` configuration.

    Returns:
        ``(o, final_state)`` where ``o`` is ``[1, T, HV, V]`` and
        ``final_state`` is the same state pool as ``initial_state``.
    """
    scale = k.shape[-1] ** -0.5

    assert q.dim() == 4 and q.shape[0] == 1, "q must be [1, T, H, K]"
    assert k.dim() == 4 and k.shape[0] == 1, "k must be [1, T, H, K]"
    assert v.dim() == 4 and v.shape[0] == 1, "v must be [1, T, HV, V]"
    _, H, K = k.shape[1], k.shape[2], k.shape[3]
    V = v.shape[-1]
    HV = v.shape[2]
    N = len(cu_seqlens) - 1
    q, stride_q_tok = _ensure_qkv_token_strided(q)
    k, stride_k_tok = _ensure_qkv_token_strided(k)
    v, stride_v_tok = _ensure_qkv_token_strided(v)
    a_raw, stride_a_tok = _ensure_gate_token_strided(a_raw)
    b_raw, stride_b_tok = _ensure_gate_token_strided(b_raw)
    BK = triton.next_power_of_2(K)
    assert K == BK, f"K={K} must be a power of 2"
    if run_config is None:
        run_config = {"BV": 8, "num_warps": 1, "num_stages": 1}
    BV = min(triton.next_power_of_2(V), run_config["BV"])
    num_warps = run_config["num_warps"]
    num_stages = run_config["num_stages"]
    NV = triton.cdiv(V, BV)

    o = q.new_empty(v.shape)
    final_state = initial_state

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)
    stride_o_tok = o.stride(1)
    assert stride_o_tok == HV * V, f"stride_o_tok={stride_o_tok} must be HV*V"
    stride_state_hv = K * V

    assert ssm_state_indices.stride(-1) == 1, "2D ssm_state_indices must have contiguous rows"
    stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    assert ssm_state_write_indices.stride(-1) == 1, "2D ssm_state_write_indices must have contiguous rows"
    stride_write_indices_seq, stride_write_indices_tok = ssm_state_write_indices.stride()

    grid = (NV, N, HV)
    _fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        ssm_state_write_indices=ssm_state_write_indices,
        num_accepted_tokens=num_accepted_tokens,
        A_log=A_log,
        dt_bias=dt_bias,
        a_raw=a_raw,
        b_raw=b_raw,
        scale=scale,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        NUM_STAGES=num_stages,
        stride_q_tok=stride_q_tok,
        stride_k_tok=stride_k_tok,
        stride_v_tok=stride_v_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_o_tok=stride_o_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_state_hv=stride_state_hv,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        stride_write_indices_seq=stride_write_indices_seq,
        stride_write_indices_tok=stride_write_indices_tok,
        SOFTPLUS_BETA=1.0,
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=num_warps,
    )
    return o, final_state


# ---------------------------------------------------------------------------
# Stride helpers
# ---------------------------------------------------------------------------


def _ensure_qkv_token_strided(x: torch.Tensor):
    if x.stride()[-2:] != (x.shape[-1], 1):
        x = x.contiguous()
    return x, x.stride(1)


def _ensure_gate_token_strided(x: torch.Tensor):
    if x.stride(1) != 1:
        x = x.contiguous()
    return x, x.stride(0)

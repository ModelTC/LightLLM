"""GDN replay decode 的 kernel 辅助。

History 张量和 cursor 在 ReplayGDNState。这里只选择 kernel 配置，并发起一层重放。
"""

from __future__ import annotations

import torch
import triton

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_decode import _decode_replay
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_autotune import _replay_config, resolve_run_config


def replay(
    state,
    linear_layer_idx: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
    reqs: torch.Tensor,
    cu_seqlens: torch.Tensor,
    run_config: dict[str, int] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """执行一层 GDN replay decode。Kernel 接收这一层的 SSM 和 history；state 提供共用的 cursor 与配置。

    q、k、v 是 [tokens, heads, dim]。cu_seqlens 与 reqs 逐行对齐，长度是请求数加一。宽度为 1 时它是 0, 1, 2, ...。
    """

    assert not state.kda, "GDN replay step does not use the KDA gate"
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must be [tokens, heads, dim]"
    assert q.shape[0] == k.shape[0] == v.shape[0], "q, k, v must cover the same tokens"
    assert cu_seqlens.numel() == reqs.numel() + 1, "cu_seqlens must cover every request"
    # 没传入时走 autotune 或默认 BV/warp/stage。传入的缺项也在这里补，不在 state 上改写。
    config = (
        _replay_config(run_config)
        if run_config is not None
        else resolve_run_config(state, "replay", q, k, v, a, b, a_log, bias, cu_seqlens)
    )
    hv, kd, vd = state.ssm_state.shape[-3:]
    assert state.num_key_heads in (q.shape[-2], hv)
    out = torch.empty_like(v)
    bv = config["BV"]
    # mtp_step == 0 时 verify_width 为 1。compute_mode=precompute 仍然投影；BV < 16 才退回逐 token。
    precompute_projections = state.compute_mode == "precompute" and bv >= 16
    # 在 Python 中解析 tile 维度；Triton JIT 中直接调用 max() 会降级为 tensor 运算，无法定义静态 shape。
    padded_verify_width = triton.next_power_of_2(state.verify_width)
    sync_mark = state.tile_sync_mark
    _decode_replay[(triton.cdiv(vd, bv), reqs.numel(), hv)](
        q,
        k,
        v,
        a,
        b,
        a_log,
        bias,
        state.ssm_state[linear_layer_idx],
        state.keys[linear_layer_idx],
        state.deltas[linear_layer_idx],
        state.gates[linear_layer_idx],
        state.raw_keys[linear_layer_idx],
        state.raw_values[linear_layer_idx],
        state.betas[linear_layer_idx],
        reqs,
        state.cursors,
        sync_mark,
        cu_seqlens,
        out,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        a.stride(0),
        b.stride(0),
        q.shape[-2],
        hv,
        state.num_key_heads,
        kd,
        vd,
        state.capacity,
        state.hold_req_id,
        triton.next_power_of_2(kd),
        bv,
        precompute_projections,
        padded_verify_width,
        False,
        state.gate_lower_bound,
        BQK=max(16, 2 * padded_verify_width),
        BR=triton.next_power_of_2(max(16, state.capacity + padded_verify_width)),
        NV=triton.cdiv(vd, bv),
        num_warps=config["num_warps"],
        num_stages=config.get("num_stages", 3),
    )
    return out, config

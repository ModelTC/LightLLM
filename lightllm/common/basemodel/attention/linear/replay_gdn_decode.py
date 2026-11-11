"""GDN replay decode 的 kernel 辅助。

History 张量和 cursor 在 ReplayGDNState。这里只选择 kernel 配置，并发起一层重放。
"""

from __future__ import annotations

import torch
import triton

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import _decode_replay
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_autotune import _replay_config, resolve_run_config


def replay(
    history,
    linear_layer_idx: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
    reqs: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    run_config: dict[str, int] | None = None,
) -> tuple[torch.Tensor, dict[str, int]]:
    """执行一层 GDN replay decode。Kernel 只接收这一层的 SSM 和 history；state 只提供共用 cursor 与配置。"""

    assert not history.kda, "GDN replay step does not use the KDA gate"
    # 没传入时走 autotune 或默认 BV/warp/stage。传入的缺项也在这里补，不在 state 上改写。
    config = (
        _replay_config(run_config)
        if run_config is not None
        else resolve_run_config(history, "replay", q, k, v, a, b, a_log, bias, cu_seqlens)
    )
    hv, kd, vd = history.ssm_state.shape[-3:]
    assert history.num_key_heads in (q.shape[-2], hv)
    axis = 1 if cu_seqlens is not None else 0
    out = torch.empty_like(v)
    bv = config["BV"]
    # mtp_step == 0 时 verify_width 为 1。compute_mode=precompute 仍然投影；BV < 16 才退回逐 token。
    precompute_projections = history.compute_mode == "precompute" and bv >= 16
    # 在 Python 中解析 tile 维度；Triton JIT 中直接调用 max() 会降级为 tensor 运算，无法定义静态 shape。
    bw = triton.next_power_of_2(history.verify_width)
    _decode_replay[(triton.cdiv(vd, bv), reqs.numel(), hv)](
        q,
        k,
        v,
        a,
        b,
        a_log,
        bias,
        history.ssm_state[linear_layer_idx],
        history.keys[linear_layer_idx],
        history.deltas[linear_layer_idx],
        history.gates[linear_layer_idx],
        history.raw_keys[linear_layer_idx],
        history.raw_values[linear_layer_idx],
        history.betas[linear_layer_idx],
        reqs,
        history.cursors,
        history.write_bases,
        cu_seqlens,
        out,
        q.stride(axis),
        k.stride(axis),
        v.stride(axis),
        a.stride(0),
        b.stride(0),
        q.shape[-2],
        hv,
        history.num_key_heads,
        kd,
        vd,
        history.capacity,
        history.hold_req_id,
        triton.next_power_of_2(kd),
        bv,
        cu_seqlens is not None,
        precompute_projections,
        bw,
        False,
        history.gate_lower_bound,
        BQK=max(16, 2 * bw),
        BR=triton.next_power_of_2(max(16, history.capacity + bw)),
        num_warps=config["num_warps"],
        num_stages=config.get("num_stages", 3),
    )
    return out, config

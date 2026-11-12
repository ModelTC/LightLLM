"""KDA replay decode。

调用方传入这一层的 SSM、history 和 cursor。这里调优并执行一层重放。
"""

from __future__ import annotations

import ast

import torch
import triton

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_decode import (
    _decode_replay,
    check_replay_sync_mark,
)
from lightllm.common.triton_utils.autotuner import AutotuneKernelType, autotune


def get_configs() -> list[dict[str, int]]:
    """返回 KDA replay 的候选 Triton 配置。"""
    return [
        {"BV": bv, "num_warps": warps, "num_stages": stages}
        for bv, warps in [(8, 1), (16, 1), (16, 2), (32, 1), (32, 2), (32, 4), (64, 2), (64, 4), (128, 4)]
        for stages in (1, 2, 3)
    ]


def get_static_key(
    state_ssm: torch.Tensor,
    state_keys: torch.Tensor,
    q: torch.Tensor,
    state_verify_width: int,
    state_compute_mode: str,
    state_gate_lower_bound: float,
) -> dict[str, object]:
    """区分头维度、状态精度、history 容量、gate 下界和投影模式。"""
    _, hv, kd, vd = state_ssm.shape
    return {
        "v": 1,
        "shape": str((q.shape[-2], hv, state_keys.shape[-3], kd, vd, state_verify_width)),
        "dtype": str((q.dtype, state_ssm.dtype)),
        "capacity": state_keys.shape[-2],
        "lower": state_gate_lower_bound,
        "recon_dtype": str(state_keys.dtype),
        "projection": "kda_mma" if state_compute_mode == "precompute" else "inline",
    }


def get_run_key(q: torch.Tensor, cu_seqlens: torch.Tensor) -> tuple[int, int]:
    """调优键是 token 数和请求数。q 是 [tokens, heads, dim]。"""
    return q.shape[0], cu_seqlens.numel() - 1


def _check_replay_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    reqs: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must be [tokens, heads, dim]"
    assert q.shape[0] == k.shape[0] == v.shape[0], "q, k, v must cover the same tokens"
    assert cu_seqlens.numel() == reqs.numel() + 1, "cu_seqlens must cover every request"


def rebuild_inputs(
    state_ssm: torch.Tensor,
    state_keys: torch.Tensor,
    state_deltas: torch.Tensor,
    state_gates: torch.Tensor,
    state_raw_keys: torch.Tensor,
    state_raw_values: torch.Tensor,
    state_betas: torch.Tensor,
    state_cursors: torch.Tensor,
    state_tile_sync_mark: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
    reqs: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_verify_width: int,
    state_compute_mode: str,
    state_gate_lower_bound: float,
    run_config: dict[str, int] | None = None,
) -> tuple[tuple, dict[str, object]]:
    """为一层 scratch 和同形状激活分配调优输入。不复制服务状态池。"""
    del state_cursors, reqs, run_config
    tokens, run_count = get_run_key(q=q, cu_seqlens=cu_seqlens)
    # 一行一个 token 时请求数就是 token 数。打包窗口再用 verify_width 折成请求数。
    batch = run_count if tokens == run_count else min(run_count, triton.cdiv(tokens, state_verify_width))
    slots = batch + 1
    _, hv, kd, vd = state_ssm.shape
    key_heads, capacity = state_keys.shape[-3], state_keys.shape[-2]
    device = q.device
    rows_per_slot = state_tile_sync_mark.shape[0] // state_ssm.shape[0]
    copied = [
        torch.empty(x.shape, device=x.device, dtype=x.dtype).normal_(0, 0.2) for x in (q, k, v, a, b, a_log, bias)
    ]
    bench_reqs = torch.cat(
        (
            torch.arange(batch, dtype=torch.int32, device=device),
            torch.full((run_count - batch,), batch, dtype=torch.int32, device=device),
        )
    )
    # 计时用均匀窗口。宽度为 1 时 verify_width 就是 1。
    cu = torch.arange(run_count + 1, dtype=torch.int32, device=device)
    cu = (cu * state_verify_width).clamp_max(tokens)
    return (), dict(
        state_ssm=torch.empty(slots, hv, kd, vd, device=device, dtype=state_ssm.dtype).normal_(0, 0.02),
        state_keys=torch.empty(slots, key_heads, capacity, kd, device=device, dtype=state_keys.dtype),
        state_deltas=torch.empty(slots, hv, capacity, vd, device=device, dtype=state_deltas.dtype),
        state_gates=torch.empty(slots, hv, capacity, kd, device=device, dtype=state_gates.dtype),
        state_raw_keys=torch.empty(slots, key_heads, capacity, kd, device=device, dtype=state_raw_keys.dtype),
        state_raw_values=torch.empty(slots, hv, capacity, vd, device=device, dtype=state_raw_values.dtype),
        state_betas=torch.empty(slots, hv, capacity, device=device, dtype=state_betas.dtype),
        state_cursors=torch.zeros(slots, dtype=torch.int32, device=device),
        state_tile_sync_mark=torch.zeros((slots * rows_per_slot, key_heads), dtype=torch.int32, device=device),
        q=copied[0],
        k=copied[1],
        v=copied[2],
        a=copied[3],
        b=copied[4],
        a_log=copied[5],
        bias=copied[6],
        reqs=bench_reqs,
        cu_seqlens=cu,
        state_verify_width=state_verify_width,
        state_compute_mode=state_compute_mode,
        state_gate_lower_bound=state_gate_lower_bound,
    )


def _run_key_distance(left: str, right: str) -> int:
    return sum(abs(x - y) for x, y in zip(ast.literal_eval(left), ast.literal_eval(right)))


@autotune(
    kernel_name="replay_kda_decode:v1",
    configs_gen_func=get_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    run_key_distance_func=_run_key_distance,
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    rebuild_input_func=rebuild_inputs,
    warmup_all_exist_config=False,
)
def replay(
    state_ssm: torch.Tensor,
    state_keys: torch.Tensor,
    state_deltas: torch.Tensor,
    state_gates: torch.Tensor,
    state_raw_keys: torch.Tensor,
    state_raw_values: torch.Tensor,
    state_betas: torch.Tensor,
    state_cursors: torch.Tensor,
    state_tile_sync_mark: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
    reqs: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_verify_width: int,
    state_compute_mode: str,
    state_gate_lower_bound: float,
    run_config: dict[str, int] | None = None,
) -> torch.Tensor:
    """执行一层 KDA replay decode。

    state_* 张量已经是这一层的视图。ssm 是 [slots, value_heads, key_dim, value_dim]，
    keys 是 [slots, key_heads, capacity, key_dim]。最后一个 slot 是 HOLD。
    q、k、v 是 [tokens, heads, dim]。cu_seqlens 与 reqs 逐行对齐，长度是请求数加一。
    宽度为 1 时它是 0, 1, 2, ...。
    """

    _check_replay_inputs(q=q, k=k, v=v, reqs=reqs, cu_seqlens=cu_seqlens)
    if run_config is None:
        run_config = {"BV": 32, "num_warps": 1, "num_stages": 3}
    _, hv, kd, vd = state_ssm.shape
    key_heads = state_keys.shape[-3]
    capacity = state_keys.shape[-2]
    assert key_heads in (q.shape[-2], hv)
    out = torch.empty_like(v)
    bv = run_config["BV"]
    # mtp_step == 0 时 verify_width 为 1。compute_mode=precompute 仍然投影；BV < 16 才退回逐 token。
    precompute_projections = state_compute_mode == "precompute" and bv >= 16
    padded_verify_width = triton.next_power_of_2(state_verify_width)
    check_replay_sync_mark(
        sync_mark=state_tile_sync_mark,
        reqs=reqs,
        value_heads=hv,
        key_heads=key_heads,
    )
    _decode_replay[(triton.cdiv(vd, bv), hv, reqs.numel())](
        Q=q,
        StepK=k,
        StepV=v,
        A=a,
        B=b,
        Alog=a_log,
        Bias=bias,
        State=state_ssm,
        Keys=state_keys,
        Deltas=state_deltas,
        Gates=state_gates,
        RawKeys=state_raw_keys,
        RawValues=state_raw_values,
        Betas=state_betas,
        Reqs=reqs,
        Cursors=state_cursors,
        SyncMark=state_tile_sync_mark,
        Cu=cu_seqlens,
        Out=out,
        SQ=q.stride(0),
        SK=k.stride(0),
        SV=v.stride(0),
        SA=a.stride(0),
        SB=b.stride(0),
        H=q.shape[-2],
        HV=hv,
        KH=key_heads,
        K=kd,
        V=vd,
        L=capacity,
        HOLD=state_ssm.shape[0] - 1,
        BK=triton.next_power_of_2(kd),
        BV=bv,
        PRECOMPUTE_PROJECTIONS=precompute_projections,
        BW=padded_verify_width,
        KDA=True,
        LOWER=state_gate_lower_bound,
        BQK=max(16, 2 * padded_verify_width),
        BR=triton.next_power_of_2(max(16, capacity + padded_verify_width)),
        NV=triton.cdiv(vd, bv),
        num_warps=run_config["num_warps"],
        num_stages=run_config["num_stages"],
    )
    return out

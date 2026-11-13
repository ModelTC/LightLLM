"""GDN replay decode。

History 张量和 cursor 在 ReplayGDNState。这里调优并执行一层重放。
"""

from __future__ import annotations

import ast

import torch
import triton

from lightllm.common.attention.backends._shared.linear.replayssm_decode import _decode_replay, check_replay_sync_mark
from lightllm.common.attention_state.bounded_att.conv_ssm_types import ConvSSMStateConfig
from lightllm.common.triton_utils.autotuner import AutotuneKernelType, autotune


def get_configs() -> list[dict[str, int]]:
    """返回 GDN replay 的候选 Triton 配置。"""
    return [
        {"BV": bv, "num_warps": warps, "num_stages": stages}
        for bv, warps in [(8, 1), (16, 1), (16, 2), (32, 1), (32, 2), (32, 4), (64, 2), (64, 4), (128, 4)]
        for stages in (1, 2, 3)
    ]


def get_static_key(state, q) -> dict[str, object]:
    """区分头维度、状态精度、history 容量和投影模式。"""
    _, _, hv, kd, vd = state.ssm_state.shape
    return {
        "v": 1,
        "shape": str((q.shape[-2], hv, state.num_key_heads, kd, vd, state.verify_width)),
        "dtype": str((q.dtype, state.ssm_state.dtype)),
        "capacity": state.capacity,
        "recon_dtype": str(state.keys.dtype),
        "projection": "gdn_mma" if state.compute_mode == "precompute" else "inline",
    }


def get_run_key(q, cu_seqlens) -> tuple[int, int]:
    """调优键是 token 数和请求数。q 是 [tokens, heads, dim]。"""
    return q.shape[0], cu_seqlens.numel() - 1


def _check_replay_inputs(q, k, v, reqs, cu_seqlens) -> None:
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must be [tokens, heads, dim]"
    assert q.shape[0] == k.shape[0] == v.shape[0], "q, k, v must cover the same tokens"
    assert cu_seqlens.numel() == reqs.numel() + 1, "cu_seqlens must cover every request"


def rebuild_inputs(state, linear_layer_idx, q, k, v, a, b, a_log, bias, reqs, cu_seqlens, run_config=None):
    """为一层 scratch 和同形状激活分配调优输入。不复制服务状态池。"""
    del linear_layer_idx, reqs, run_config
    tokens, run_count = get_run_key(q, cu_seqlens)
    # 一行一个 token 时请求数就是 token 数。打包窗口再用 verify_width 折成请求数。
    batch = run_count if tokens == run_count else min(run_count, triton.cdiv(tokens, state.verify_width))
    _, _, hv, kd, vd = state.ssm_state.shape
    scratch = type(state)(
        ConvSSMStateConfig(
            logical_layer_ids=(0,),
            global_k_heads=state.num_key_heads,
            global_v_heads=hv,
            key_dim=kd,
            value_dim=vd,
            conv_width=1,
            conv_dtype=q.dtype,
            ssm_dtype=state.ssm_state.dtype,
        ),
        batch + 1,
        mtp_step=state.verify_width - 1,
        device=q.device,
        ssm_replay_config={"cache_len": state.capacity, "compute_mode": state.compute_mode},
        gate_lower_bound=state.gate_lower_bound,
    )
    scratch.ssm_state.normal_(0, 0.02)
    copied = [
        torch.empty(x.shape, device=x.device, dtype=x.dtype).normal_(0, 0.2) for x in (q, k, v, a, b, a_log, bias)
    ]
    bench_reqs = torch.cat(
        (
            torch.arange(batch, dtype=torch.int32, device=q.device),
            torch.full((run_count - batch,), batch, dtype=torch.int32, device=q.device),
        )
    )
    # 计时用均匀窗口。宽度为 1 时 verify_width 就是 1。
    cu = torch.arange(run_count + 1, dtype=torch.int32, device=q.device)
    cu = (cu * state.verify_width).clamp_max(tokens)
    return (), dict(
        state=scratch,
        linear_layer_idx=0,
        q=copied[0],
        k=copied[1],
        v=copied[2],
        a=copied[3],
        b=copied[4],
        a_log=copied[5],
        bias=copied[6],
        reqs=bench_reqs,
        cu_seqlens=cu,
    )


def _run_key_distance(left: str, right: str) -> int:
    return sum(abs(x - y) for x, y in zip(ast.literal_eval(left), ast.literal_eval(right)))


@autotune(
    kernel_name="replay_gdn_decode:v1",
    configs_gen_func=get_configs,
    static_key_func=get_static_key,
    run_key_func=get_run_key,
    run_key_distance_func=_run_key_distance,
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    rebuild_input_func=rebuild_inputs,
    warmup_all_exist_config=False,
)
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
) -> torch.Tensor:
    """执行一层 GDN replay decode。Kernel 接收这一层的 SSM 和 history；state 提供共用的 cursor 与配置。

    q、k、v 是 [tokens, heads, dim]。cu_seqlens 与 reqs 逐行对齐，长度是请求数加一。宽度为 1 时它是 0, 1, 2, ...。
    """

    assert not state.kda, "GDN replay step does not use the KDA gate"
    _check_replay_inputs(q, k, v, reqs, cu_seqlens)
    if run_config is None:
        run_config = {"BV": 32, "num_warps": 1, "num_stages": 3}
    hv, kd, vd = state.ssm_state.shape[-3:]
    assert state.num_key_heads in (q.shape[-2], hv)
    out = torch.empty_like(v)
    bv = run_config["BV"]
    # mtp_step == 0 时 verify_width 为 1。compute_mode=precompute 仍然投影；BV < 16 才退回逐 token。
    precompute_projections = state.compute_mode == "precompute" and bv >= 16
    padded_verify_width = triton.next_power_of_2(state.verify_width)
    check_replay_sync_mark(
        sync_mark=state.tile_sync_mark,
        reqs=reqs,
        value_heads=hv,
        key_heads=state.num_key_heads,
    )
    _decode_replay[(triton.cdiv(vd, bv), hv, reqs.numel())](
        Q=q,
        StepK=k,
        StepV=v,
        A=a,
        B=b,
        Alog=a_log,
        Bias=bias,
        State=state.ssm_state[linear_layer_idx],
        Keys=state.keys[linear_layer_idx],
        Deltas=state.deltas[linear_layer_idx],
        Gates=state.gates[linear_layer_idx],
        RawKeys=state.raw_keys[linear_layer_idx],
        RawValues=state.raw_values[linear_layer_idx],
        Betas=state.betas[linear_layer_idx],
        Reqs=reqs,
        Cursors=state.cursors,
        SyncMark=state.tile_sync_mark,
        Cu=cu_seqlens,
        Out=out,
        SQ=q.stride(0),
        SK=k.stride(0),
        SV=v.stride(0),
        SA=a.stride(0),
        SB=b.stride(0),
        H=q.shape[-2],
        HV=hv,
        KH=state.num_key_heads,
        K=kd,
        V=vd,
        L=state.capacity,
        HOLD=state.hold_req_id,
        BK=triton.next_power_of_2(kd),
        BV=bv,
        PRECOMPUTE_PROJECTIONS=precompute_projections,
        BW=padded_verify_width,
        KDA=False,
        LOWER=state.gate_lower_bound,
        BQK=max(16, 2 * padded_verify_width),
        BR=triton.next_power_of_2(max(16, state.capacity + padded_verify_width)),
        NV=triton.cdiv(vd, bv),
        num_warps=run_config["num_warps"],
        num_stages=run_config["num_stages"],
    )
    return out

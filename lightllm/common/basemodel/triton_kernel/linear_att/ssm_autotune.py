"""使用共享 autotuner 调优单层 SSM update cycle。"""

from __future__ import annotations

import ast

import torch
import triton

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_cursor import advance_cursor
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType, autotune
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_DEFAULT_REPLAY_CONFIG = {"BV": 32, "num_warps": 1, "num_stages": 3}


def resolve_run_config(state, mode, q, k, v, a, b, a_log, bias, cu_seqlens) -> dict[str, int]:
    """返回当前 batch 的固定或已调优 SSM kernel 配置。"""
    if state._config_is_fixed and state.run_config is not None:
        return _replay_config(state.run_config)
    default = _replay_config(state.run_config)
    key = (tuple(static_key(state, mode, q, cu_seqlens).items()), run_key(q, cu_seqlens))
    if key in state.batch_configs:
        return state.batch_configs[key]
    if not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        return default
    config = _replay_config(select_config(state, mode, q, k, v, a, b, a_log, bias, cu_seqlens))
    # 每个 captured graph 保留自己的布局，即使其他 bucket 后续完成 warmup。
    state.batch_configs[key] = config
    logger.info(f"SSM {mode} update-cycle config: {config}, warmup batch: {key[1]}")
    return config


def get_configs() -> list[dict[str, int]]:
    """返回 ReplaySSM update cycle 的候选 Triton 配置。"""
    return [
        {"BV": bv, "num_warps": warps, "num_stages": stages}
        for bv, warps in [(8, 1), (16, 1), (16, 2), (32, 1), (32, 2), (32, 4), (64, 2), (64, 4), (128, 4)]
        for stages in (1, 2, 3)
    ]


def static_key(state, mode, q, cu_seqlens) -> dict[str, object]:
    """构造区分状态布局、数据类型和投影模式的静态调优键。"""
    _, _, hv, kd, vd = state.ssm_state.shape
    # 保持文件名较短，同时区分 packed projection、TP 布局、状态精度、递推规则和 history 容量。
    key = {
        "mode": mode,
        "v": 10,
        "shape": str((q.shape[-2], hv, state.num_key_heads, kd, vd, state.verify_width)),
        "dtype": str((q.dtype, state.ssm_state.dtype)),
        "capacity": state.capacity,
        "kda": int(state.kda),
    }
    if state.kda:
        key["lower"] = state.gate_lower_bound
    key["recon_dtype"] = str(state.keys.dtype)
    key["projection"] = ("kda_mma" if state.kda else "gdn_mma") if state.compute_mode == "precompute" else "inline"
    return key


def run_key(q, cu_seqlens) -> tuple[int, int]:
    """返回调优键：token 数，以及请求数。q 是 [tokens, heads, dim]。没传前缀和时一行一个请求。"""
    tokens = q.shape[0]
    return tokens, cu_seqlens.numel() - 1 if cu_seqlens is not None else tokens


def rebuild_inputs(
    state, mode, q, k, v, a, b, a_log, bias, cu_seqlens=None, workload=None, run_config=None
) -> tuple[tuple[object, ...], dict[str, object]]:
    """为一层 scratch 和同形状的紧凑激活分配调优输入。不复制服务状态池。"""
    tokens, run_count = run_key(q, cu_seqlens)
    # 一行一个 token 时请求数就是 token 数。打包窗口再用 verify_width 折成请求数。
    batch = run_count if tokens == run_count else min(run_count, triton.cdiv(tokens, state.verify_width))
    _, _, hv, kd, vd = state.ssm_state.shape
    from lightllm.common.attention_state.bounded_att.conv_ssm_types import ConvSSMStateConfig

    # 计时只需要一层、本 batch 的槽位。正式执行仍用调用方原来的 state。
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
    # benchmark 回调不能再次触发配置选择。
    scratch._config_is_fixed = True
    inputs = [
        torch.empty(x.shape, device=x.device, dtype=x.dtype).normal_(0, 0.2) for x in (q, k, v, a, b, a_log, bias)
    ]
    reqs = torch.cat(
        (
            torch.arange(batch, dtype=torch.int32, device=q.device),
            torch.full((run_count - batch,), batch, dtype=torch.int32, device=q.device),
        )
    )
    # 计时用均匀窗口，不读调用方的前缀和。宽度为 1 时 verify_width 就是 1。
    cu = torch.arange(run_count + 1, dtype=torch.int32, device=q.device)
    cu = (cu * state.verify_width).clamp_max(tokens)
    return (), dict(
        state=scratch,
        mode=mode,
        q=inputs[0],
        k=inputs[1],
        v=inputs[2],
        a=inputs[3],
        b=inputs[4],
        a_log=inputs[5],
        bias=inputs[6],
        cu_seqlens=cu,
        workload=(reqs,),
    )


def replay_layer(state, *args, **kwargs):
    """按 state 的具体类型调用对应的 replay decode。"""

    from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_state import ReplayGDNState
    from lightllm.common.attention_state.bounded_att.kda.replay_kda_state import ReplayKDAState

    if isinstance(state, ReplayKDAState):
        from lightllm.common.basemodel.attention.linear.replay_kda_decode import replay
    elif isinstance(state, ReplayGDNState):
        from lightllm.common.basemodel.attention.linear.replay_gdn_decode import replay
    else:
        raise TypeError(f"replay_layer expects ReplayGDNState or ReplayKDAState, got {type(state).__name__}")
    return replay(state, *args, **kwargs)


@autotune(
    kernel_name="ssm_update_cycle:v2",
    configs_gen_func=get_configs,
    static_key_func=static_key,
    run_key_func=run_key,
    run_key_distance_func=lambda a, b: sum(abs(x - y) for x, y in zip(ast.literal_eval(a), ast.literal_eval(b))),
    kernel_type=AutotuneKernelType.DECODE_ATTENTION,
    rebuild_input_func=rebuild_inputs,
    warmup_all_exist_config=False,
)
def select_config(
    state, mode, q, k, v, a, b, a_log, bias, cu_seqlens=None, workload=None, run_config=None
) -> dict[str, int] | None:
    """搜索 update cycle 布局；计时只使用重建的临时输入。"""
    if workload is None:
        return run_config
    state.run_config = run_config
    (reqs,) = workload
    state.cursors.zero_()
    # 把前缀写满但不越出 capacity。replay kernel 自己读 cursor。调优窗口全部接受。
    tokens = run_key(q, cu_seqlens)[0]
    rounds = max(1, state.capacity // min(state.verify_width, tokens))
    for _ in range(rounds):
        replay_layer(state, 0, q, k, v, a, b, a_log, bias, reqs, cu_seqlens)
        advance_cursor(state, reqs, cu_seqlens)
    return run_config


def _replay_config(config: dict | None) -> dict[str, int]:
    if not config:
        return dict(_DEFAULT_REPLAY_CONFIG)
    return {
        "BV": int(config["BV"]),
        "num_warps": int(config["num_warps"]),
        "num_stages": int(config.get("num_stages", 3)),
    }

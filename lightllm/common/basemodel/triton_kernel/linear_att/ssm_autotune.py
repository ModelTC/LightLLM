"""使用共享 autotuner 调优单层 SSM update cycle。"""

from __future__ import annotations

import ast

import torch
import triton

from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType, autotune
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

_DEFAULT_REPLAY_CONFIG = {"BV": 32, "num_warps": 1, "num_stages": 3}


def resolve_run_config(ssm_backend, mode, q, k, v, a, b, a_log, bias, cu_seqlens) -> dict[str, int]:
    """返回当前 batch 的固定或已调优 SSM kernel 配置。"""
    if ssm_backend._config_is_fixed and ssm_backend.run_config is not None:
        return _replay_config(ssm_backend.run_config)
    default = _replay_config(ssm_backend.run_config)
    key = (tuple(static_key(ssm_backend, mode, q, cu_seqlens).items()), run_key(q, cu_seqlens))
    if key in ssm_backend.batch_configs:
        return ssm_backend.batch_configs[key]
    if not Autotuner.is_kernel_autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        return default
    config = _replay_config(select_config(ssm_backend, mode, q, k, v, a, b, a_log, bias, cu_seqlens))
    # 每个 captured graph 保留自己的布局，即使其他 bucket 后续完成 warmup。
    ssm_backend.batch_configs[key] = config
    logger.info(f"SSM {mode} update-cycle config: {config}, warmup batch: {key[1]}")
    return config


def get_configs() -> list[dict[str, int]]:
    """返回 ReplaySSM update cycle 的候选 Triton 配置。"""
    return [
        {"BV": bv, "num_warps": warps, "num_stages": stages}
        for bv, warps in [(8, 1), (16, 1), (16, 2), (32, 1), (32, 2), (32, 4), (64, 2), (64, 4), (128, 4)]
        for stages in (1, 2, 3)
    ]


def static_key(ssm_backend, mode, q, cu_seqlens) -> dict[str, object]:
    """构造区分状态布局、数据类型和投影模式的静态调优键。"""
    _, _, hv, kd, vd = ssm_backend.ssm_state.shape
    # 保持文件名较短，同时区分 packed projection、TP 布局、状态精度、递推规则和 history 容量。
    key = {
        "mode": mode,
        "v": 7,
        "shape": str((q.shape[-2], hv, ssm_backend.num_key_heads, kd, vd, ssm_backend.verify_width)),
        "dtype": str((q.dtype, ssm_backend.ssm_state.dtype)),
        "history": ssm_backend.capacity,
        "gate": int(ssm_backend.kda),
        "varlen": cu_seqlens is not None,
    }
    if ssm_backend.kda:
        key["lower"] = ssm_backend.gate_lower_bound
    key["history_dtype"] = str(ssm_backend.keys.dtype)
    key["projection"] = (
        ("kda_mma" if ssm_backend.kda else "gdn_mma") if ssm_backend.compute_mode == "precompute" else "inline"
    )
    return key


def run_key(q, cu_seqlens) -> tuple[int, int]:
    """返回调优使用的 token 数和请求数键。"""
    tokens = q.shape[1 if cu_seqlens is not None else 0]
    return tokens, cu_seqlens.numel() - 1 if cu_seqlens is not None else tokens


def rebuild_inputs(
    ssm_backend, mode, q, k, v, a, b, a_log, bias, cu_seqlens=None, workload=None, run_config=None
) -> tuple[tuple[object, ...], dict[str, object]]:
    """为一层 scratch 和同形状的紧凑激活分配调优输入。不复制服务状态池。"""
    tokens, sequences = run_key(q, cu_seqlens)
    batch = min(sequences, triton.cdiv(tokens, ssm_backend.verify_width)) if cu_seqlens is not None else tokens
    _, _, hv, kd, vd = ssm_backend.ssm_state.shape
    from lightllm.common.attention_state.bounded_att.conv_ssm_types import ConvSSMStateConfig

    # 计时只需要一层、本 batch 的槽位。正式执行仍用调用方原来的 state。
    scratch = type(ssm_backend)(
        ConvSSMStateConfig(
            logical_layer_ids=(0,),
            global_k_heads=ssm_backend.num_key_heads,
            global_v_heads=hv,
            key_dim=kd,
            value_dim=vd,
            conv_width=1,
            conv_dtype=q.dtype,
            ssm_dtype=ssm_backend.ssm_state.dtype,
        ),
        batch + 1,
        mtp_step=ssm_backend.verify_width - 1,
        device=q.device,
        ssm_replay_config={"cache_len": ssm_backend.capacity, "compute_mode": ssm_backend.compute_mode},
        gate_lower_bound=ssm_backend.gate_lower_bound,
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
            torch.full((sequences - batch,), batch, dtype=torch.int32, device=q.device),
        )
    )
    cu = (torch.arange(sequences + 1, dtype=torch.int32, device=q.device) * ssm_backend.verify_width).clamp_max(tokens)
    return (), dict(
        ssm_backend=scratch,
        mode=mode,
        q=inputs[0],
        k=inputs[1],
        v=inputs[2],
        a=inputs[3],
        b=inputs[4],
        a_log=inputs[5],
        bias=inputs[6],
        cu_seqlens=cu if cu_seqlens is not None else None,
        workload=(reqs,),
    )


def replay_layer(history, *args, **kwargs):
    """按 history 的具体类型调用对应的 replay decode。"""

    from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_state import ReplayGDNState
    from lightllm.common.attention_state.bounded_att.kda.replay_kda_state import ReplayKDAState

    if isinstance(history, ReplayKDAState):
        from lightllm.common.basemodel.attention.linear.replay_kda_decode import replay
    elif isinstance(history, ReplayGDNState):
        from lightllm.common.basemodel.attention.linear.replay_gdn_decode import replay
    else:
        raise TypeError(f"replay_layer expects ReplayGDNState or ReplayKDAState, got {type(history).__name__}")
    return replay(history, *args, **kwargs)


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
    ssm_backend, mode, q, k, v, a, b, a_log, bias, cu_seqlens=None, workload=None, run_config=None
) -> dict[str, int] | None:
    """搜索 update cycle 布局；计时只使用重建的临时输入。"""
    if workload is None:
        return run_config
    ssm_backend.run_config = run_config
    (reqs,) = workload
    ssm_backend.cursors.zero_()
    # 把前缀写满但不越出 capacity。replay kernel 自己读 cursor。调优窗口全部接受。
    tokens = q.shape[1 if cu_seqlens is not None else 0]
    rounds = max(1, ssm_backend.capacity // min(ssm_backend.verify_width, tokens))
    for _ in range(rounds):
        replay_layer(ssm_backend, 0, q, k, v, a, b, a_log, bias, reqs, cu_seqlens)
        adopt_cu = cu_seqlens
        if adopt_cu is None:
            adopt_cu = torch.arange(reqs.numel() + 1, dtype=torch.int32, device=reqs.device)
        ssm_backend.advance_cursor(reqs, adopt_cu)
    return run_config


def _replay_config(config: dict | None) -> dict[str, int]:
    if not config:
        return dict(_DEFAULT_REPLAY_CONFIG)
    return {
        "BV": int(config["BV"]),
        "num_warps": int(config["num_warps"]),
        "num_stages": int(config.get("num_stages", 3)),
    }

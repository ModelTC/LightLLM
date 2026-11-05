import torch


def create_ssm_executor(config, request_capacity, mtp_step, device, mode, replay_config, *, kda, gate_lower_bound):
    """按统一 state layout 创建 native、compact 或 replay executor。

    三种模式共享 canonical SSM state；差异只在 decode 中间量和接受阶段的
    state 更新方式。native 为每个 MTP 位置分配物理槽位，compact/replay
    则为每个请求分配一个槽位并在各自 executor 内保存临时数据。
    """

    if mode not in ("native", "compact", "replay"):
        raise ValueError(f"unknown SSM state mode: {mode}")
    slots = mtp_step + 1 if mode == "native" else 1
    state = torch.zeros(
        (len(config.logical_layer_ids), request_capacity * slots, config.num_v_heads, config.key_dim, config.value_dim),
        dtype=config.ssm_dtype,
        device=device,
    )
    kwargs = dict(kda=kda, gate_lower_bound=gate_lower_bound)
    if mode == "native":
        from .native import NativeSSMExecutor

        return NativeSSMExecutor(state, mtp_step + 1, **kwargs)
    kwargs.update(num_key_heads=config.num_k_heads)
    if mode == "compact":
        from .compact import CompactSSMExecutor

        return CompactSSMExecutor(state, mtp_step + 1, config.conv_dtype, **kwargs)
    from .replay import ReplaySSMExecutor

    replay_config = replay_config or {}
    return ReplaySSMExecutor(
        state,
        replay_config.get("cache_len", 16),
        mtp_step + 1,
        config.conv_dtype,
        compute_mode=replay_config.get("compute_mode", "inline"),
        fold_method=replay_config.get("fold_method", "raw"),
        **kwargs,
    )

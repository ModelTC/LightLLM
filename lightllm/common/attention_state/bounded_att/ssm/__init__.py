import torch


def create_ssm_state(config, request_capacity, mtp_step, device, mode, replay_config, *, kda, gate_lower_bound):
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
        from .native import NativeSSMState

        return NativeSSMState(state, mtp_step + 1, **kwargs)
    kwargs.update(num_key_heads=config.num_k_heads)
    if mode == "compact":
        from .compact import CompactSSMState

        return CompactSSMState(state, mtp_step + 1, config.conv_dtype, **kwargs)
    from .replay import ReplaySSMState

    replay_config = replay_config or {}
    return ReplaySSMState(
        state,
        replay_config.get("cache_len", 16),
        mtp_step + 1,
        config.conv_dtype,
        compute_mode=replay_config.get("compute_mode", "inline"),
        fold_method=replay_config.get("fold_method", "raw"),
        **kwargs,
    )

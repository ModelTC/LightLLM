"""测试用的 replay history。正式路径走 ReplayGDNState / ReplayKDAState 的构造函数。"""

import torch

from lightllm.common.attention_state.bounded_att.gdn.replay_gdn_state import ReplayGDNState
from lightllm.common.attention_state.bounded_att.kda.replay_kda_state import ReplayKDAState


def attach_replay_history(
    ssm: torch.Tensor,
    capacity: int,
    verify_width: int,
    activation_dtype: torch.dtype = torch.bfloat16,
    num_key_heads: int | None = None,
    gate_lower_bound: float = -5.0,
    compute_mode: str = "inline",
    run_config: dict[str, int] | None = None,
    *,
    kda: bool = False,
):
    """把连续前缀挂到调用方已有的 SSM 上，保留它原来的 stride。

    kernel 测试需要非连续的层距，不能再走会重新分配状态的构造函数。
    """

    # capacity 是 2 的幂，并且不小于一次验证窗口。
    assert ssm.dtype in (torch.float32, torch.bfloat16), "replay requires FP32 or BF16 SSM state"
    assert verify_width > 0 and capacity >= max(4, verify_width) and capacity & (capacity - 1) == 0
    assert compute_mode in ("inline", "precompute")
    state_cls = ReplayKDAState if kda else ReplayGDNState
    owner = state_cls.__new__(state_cls)
    owner.config = None
    owner.state_name = None
    owner.request_capacity = ssm.shape[1]
    owner.mtp_step = verify_width - 1
    owner._algorithm = "kda" if kda else "gdn"
    owner._logical_to_storage_layer = {}
    owner.conv_state = None
    owner.mtp_state_index = None
    owner.big_pages_pool = None
    owner.small_pages_pool = None
    owner.ssm_state = ssm
    owner.kda = kda
    owner.gate_lower_bound = gate_lower_bound
    owner.capacity = capacity
    owner.verify_width = verify_width
    owner.compute_mode = compute_mode
    owner.run_config = run_config
    owner._config_is_fixed = run_config is not None
    owner.batch_configs = {}
    layers, slots, hv, k, v = ssm.shape
    owner.num_key_heads = hv if num_key_heads is None else num_key_heads
    assert owner.num_key_heads > 0 and hv % owner.num_key_heads == 0
    # 最后一个 slot 是 HOLD。cursor 是尚未折回的已接受长度，write_bases 是本步写入起点。
    owner.hold_req_id = slots - 1
    owner.cursors = torch.zeros(slots, dtype=torch.int32, device=ssm.device)
    owner.write_bases = torch.zeros(slots, dtype=torch.int32, device=ssm.device)
    # 重建用 k/d：bf16 激活用 fp16。折回仍用原始 k/v。KDA 的 g 按 key 通道存。
    reconstruction_dtype = torch.float16 if activation_dtype == torch.bfloat16 else torch.float32
    key_shape = (layers, slots, owner.num_key_heads, capacity, k)
    gate_shape = (layers, slots, hv, capacity, k) if kda else (layers, slots, hv, capacity)
    owner.keys = torch.empty(key_shape, dtype=reconstruction_dtype, device=ssm.device)
    owner.deltas = torch.empty((layers, slots, hv, capacity, v), dtype=reconstruction_dtype, device=ssm.device)
    owner.gates = torch.empty(gate_shape, dtype=torch.float32, device=ssm.device)
    owner.raw_keys = torch.empty(key_shape, dtype=activation_dtype, device=ssm.device)
    owner.raw_values = torch.empty((layers, slots, hv, capacity, v), dtype=activation_dtype, device=ssm.device)
    owner.betas = torch.empty((layers, slots, hv, capacity), dtype=torch.float32, device=ssm.device)
    return owner

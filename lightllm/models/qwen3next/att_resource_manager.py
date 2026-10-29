"""Qwen3Next attention resources with explicit model-owned State objects."""

from __future__ import annotations

from typing import TYPE_CHECKING
from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import StatePageAllocator
from lightllm.common.att_resources.bounded_att import create_gdn_state, ConvSSMStateConfig

if TYPE_CHECKING:
    import torch
    from collections.abc import Sequence
    from lightllm.common.att_resources.allocators.full_att_kv_allocator import FullAttKVAllocator
    from lightllm.common.att_resources.full_att.base_state import FullAttentionState
    from lightllm.models.qwen3next.att_resource_config import Qwen3NextResourceConfig


class Qwen3NextAttResourceManager(AttResourceManager):
    # ========================================================================
    # 一、模型资源：创建 State 与绑定存储
    # ========================================================================

    def __init__(
        self,
        cache_config: Qwen3NextResourceConfig,
        request_capacity: int,
        mtp_step: int,
        *,
        tp_rank: int = 0,
        device: str | torch.device | int = "cuda",
        role: str = "main",
        layer_ids: Sequence[int] | None = None,
    ):
        super().__init__(request_capacity=request_capacity)
        self.cache_config = cache_config
        layers = (
            tuple(i for i in range(cache_config.all_layer_num) if (i + 1) % cache_config.full_attention_interval)
            if layer_ids is None
            else tuple(layer_ids)
        )
        config = ConvSSMStateConfig(
            layer_ids=layers,
            global_k_heads=cache_config.global_linear_k_heads,
            global_v_heads=cache_config.global_linear_v_heads,
            key_dim=cache_config.head_linear_k_dim,
            value_dim=cache_config.head_linear_v_dim,
            conv_width=cache_config.conv_kernel_size - 1,
            conv_dtype=cache_config.conv_state_dtype,
            ssm_dtype=cache_config.ssm_state_dtype,
            tp_rank=tp_rank,
            tp_world_size=cache_config.tp_world_size,
        )
        if layers:
            state = create_gdn_state(config=config, request_capacity=request_capacity, mtp_step=mtp_step, device=device)
            self.set_att_state(role, state)

    def bind_storage(
        self,
        full_state: FullAttentionState,
        full_allocator: FullAttKVAllocator,
        *,
        big_page_capacity: int,
        small_page_capacity: int,
        draft_full_state: FullAttentionState | None = None,
    ):
        """按已确定的 KV 容量绑定存储，创建本模型的大小页池与共享分配器。"""
        assert self.full_att_allocator is None, "attention storage is already bound"
        assert all(
            layer < self.cache_config.all_layer_num for layer in full_state.config.layer_ids
        ), "main State must only contain main layers"
        assert (draft_full_state is not None) == bool(
            self.cache_config.draft_full_att_kv_layer_num
        ), "draft State must match the model configuration"
        if draft_full_state is not None:
            assert all(
                layer >= self.cache_config.all_layer_num for layer in draft_full_state.config.layer_ids
            ), "draft State must only contain draft layers"

        state = self.main_gdn_state
        state.initialize_page_pools(
            big_page_capacity=big_page_capacity, small_page_capacity=small_page_capacity,
            pin_memory=state.conv_state.is_cuda,
        )
        big_allocator = StatePageAllocator(big_page_capacity, reserved_slots=2) if big_page_capacity else None
        small_allocator = StatePageAllocator(small_page_capacity) if small_page_capacity else None
        self.set_att_state("main", full_state)
        if draft_full_state is not None:
            self.set_att_state("draft0", draft_full_state)
        self.full_att_allocator = full_allocator
        self.bounded_att_big_page_allocator = big_allocator
        self.bounded_att_small_page_allocator = small_allocator
        self.bounded_att_cpu_load_big_page_slot = big_page_capacity - 2 if big_allocator else None
        self.bounded_att_cpu_offload_big_page_slot = big_page_capacity - 1 if big_allocator else None
        self.finish_state_creation()

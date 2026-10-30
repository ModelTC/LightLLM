"""Glm5Next attention resources with explicit model-owned State objects."""

from __future__ import annotations

from typing import TYPE_CHECKING
from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import StatePageAllocator
from lightllm.common.att_resources.bounded_att import create_kda_state, ConvSSMStateConfig
from lightllm.common.att_resources.att_tail import create_kpool_state, KPoolStateConfig

if TYPE_CHECKING:
    import torch
    from collections.abc import Sequence
    from lightllm.common.att_resources.allocators.full_att_kv_allocator import FullAttKVAllocator
    from lightllm.common.att_resources.full_att.base_state import FullAttentionState
    from lightllm.models.glm5_next.att_resource_config import Glm5NextResourceConfig
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel


def init_model_att_tail(model: Glm5NextTpPartModel) -> None:
    """按当前模型自己的 full-attention 层创建 tail，直接加入共享管理器。"""
    from lightllm.common.basemodel.basemodel_utils import get_full_attention_options, get_resource_config

    config = get_resource_config(model.config, model_class=type(model))
    options = get_full_attention_options(
        config, model.args, model_class=type(model),
        model_logical_layer_ids=tuple(layer.layer_num_ for layer in model.layers_infer),
    )
    if not options["logical_layer_groups"]:
        return
    cfg = model.linear_config
    state = create_kpool_state(
        config=KPoolStateConfig(
            logical_layer_ids=tuple(
                logical_layer_id for logical_layer_group in options["logical_layer_groups"]
                for logical_layer_id in logical_layer_group
            ), pool_size=cfg.index_kpool,
            head_dim=cfg.index_head_dim, dtype=cfg.full_att_dtype,
            tp_rank=model.tp_rank_, tp_world_size=model.tp_world_size_,
        ),
        request_capacity=model.max_req_num + 1, mtp_step=model.args.mtp_step, device="cuda",
    )
    model.att_resource_manager.set_att_state(model.att_resource_role, state)


class Glm5NextAttResourceManager(AttResourceManager):
    # ========================================================================
    # 一、模型资源：创建 State 与绑定存储
    # ========================================================================

    def __init__(
        self,
        cache_config: Glm5NextResourceConfig,
        request_capacity: int,
        mtp_step: int,
        *,
        tp_rank: int = 0,
        device: str | torch.device | int = "cuda",
        role: str = "main",
        bounded_logical_layer_ids: Sequence[int] | None = None,
        full_att_logical_layer_ids: Sequence[int] | None = None,
    ):
        super().__init__(request_capacity=request_capacity)
        self.cache_config = cache_config
        self.cpu_tail_restore = True
        logical_layer_ids = (
            tuple(i for i in range(cache_config.all_layer_num) if (i + 1) % cache_config.full_attention_interval)
            if bounded_logical_layer_ids is None
            else tuple(bounded_logical_layer_ids)
        )
        config = ConvSSMStateConfig(
            logical_layer_ids=logical_layer_ids,
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
        if logical_layer_ids:
            state = create_kda_state(config=config, request_capacity=request_capacity, mtp_step=mtp_step, device=device)
            self.set_att_state(role, state)
        self.att_tail_type = f"indexer_kpool_c{cache_config.index_kpool}_tail"
        own_full_att_logical_layer_ids = (
            tuple(
                range(
                    cache_config.full_attention_interval - 1,
                    cache_config.all_layer_num,
                    cache_config.full_attention_interval,
                )
            )
            if full_att_logical_layer_ids is None
            else tuple(full_att_logical_layer_ids)
        )
        tail_logical_layers_by_role = [(role, own_full_att_logical_layer_ids)]
        if role == "main" and cache_config.draft_full_att_kv_layer_num:
            tail_logical_layers_by_role.append(
                (
                    "draft0",
                    tuple(
                        range(
                            cache_config.all_layer_num,
                            cache_config.all_layer_num + cache_config.draft_full_att_kv_layer_num,
                        )
                    ),
                )
            )
        for role, logical_layer_ids in tail_logical_layers_by_role:
            if logical_layer_ids:
                state = create_kpool_state(
                    config=KPoolStateConfig(
                        logical_layer_ids=logical_layer_ids,
                        pool_size=cache_config.index_kpool,
                        head_dim=cache_config.index_head_dim,
                        dtype=cache_config.full_att_dtype,
                        tp_rank=tp_rank,
                        tp_world_size=cache_config.tp_world_size,
                    ),
                    request_capacity=request_capacity, mtp_step=mtp_step, device=device,
                )
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
            logical_layer_id < self.cache_config.all_layer_num
            for logical_layer_group in full_state.config.logical_layer_groups for logical_layer_id in logical_layer_group
        ), "main State must only contain main layers"
        assert (draft_full_state is not None) == bool(
            self.cache_config.draft_full_att_kv_layer_num
        ), "draft State must match the model configuration"
        if draft_full_state is not None:
            assert all(
                logical_layer_id >= self.cache_config.all_layer_num
                for logical_layer_group in draft_full_state.config.logical_layer_groups for logical_layer_id in logical_layer_group
            ), "draft State must only contain draft layers"

        state = self.main_kda_state
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

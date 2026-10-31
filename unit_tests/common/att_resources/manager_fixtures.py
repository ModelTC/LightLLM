"""Concrete main/draft composition used to exercise manager transfer operations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import replace
from functools import lru_cache
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.allocators import StatePageAllocator, FullAttKVAllocator
from lightllm.common.attention_state.full_att import MLAAttentionConfig, MLAFullAttentionState
from lightllm.common.attention_state.bounded_att import GDNState
from unit_tests.common.att_resources.test_full_state import full_state
from unit_tests.common.att_resources.test_bounded_state import make_state

if TYPE_CHECKING:
    import torch


class MainDraftTestResourceManager(AttentionStateManager):
    def __init__(self, device: str | torch.device | int = "cpu"):
        super().__init__(request_capacity=3)
        self.register_state("main", make_state(device=device))
        draft_state = GDNState(
            config=replace(self.main_kda_conv_fp32_ssm_fp32_state.config, global_v_heads=2),
            request_capacity=3,
            mtp_step=self.main_kda_conv_fp32_ssm_fp32_state.mtp_step,
            device=device,
        )
        draft_state.initialize_page_pools(big_page_capacity=3, small_page_capacity=2, pin_memory=device != "cpu")
        self.register_state("draft0", draft_state)
        self.register_state("main", full_state(device=device))
        prepared_full_state = MLAFullAttentionState(
            layout_config=MLAAttentionConfig(
                logical_layer_groups=self.main_gqa_fp32_state.config.logical_layer_groups,
                latent_dim=self.main_gqa_fp32_state.config.head_dim,
                dtype=self.main_gqa_fp32_state.config.dtype,
            ),
            page_size=self.main_gqa_fp32_state.page_size, device=device,
        )
        prepared_full_state.allocate_storage(self.main_gqa_fp32_state.capacity)
        self.register_state("draft0", prepared_full_state)
        self.full_att_allocator = FullAttKVAllocator(12, page_size=4, pin_memory=False)
        self.bounded_att_big_page_allocator = StatePageAllocator(3)
        self.bounded_att_small_page_allocator = StatePageAllocator(2)
        self.finish_state_registration()

    @lru_cache(maxsize=None)
    def get_cpu_page_layouts(self):
        raise NotImplementedError("This test manager uses explicitly supplied CPU layouts")

    @lru_cache(maxsize=None)
    def get_pd_token_page_layouts(self):
        raise NotImplementedError("This test manager uses explicitly supplied PD layouts")

    @lru_cache(maxsize=None)
    def get_pd_runtime_page_layouts(self):
        raise NotImplementedError("This test manager uses explicitly supplied PD layouts")

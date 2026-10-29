"""Concrete main/draft composition used to exercise manager transfer operations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import replace
from functools import lru_cache
from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.allocators import StatePageAllocator, FullAttKVAllocator
from lightllm.common.att_resources.full_att import MLAFullAttentionState
from lightllm.common.att_resources.bounded_att import GDNState
from lightllm.common.att_resources.att_tail import KPoolAttentionTailState
from unit_tests.common.att_resources.test_full_state import full_state
from unit_tests.common.att_resources.test_bounded_state import make_state
from unit_tests.common.att_resources.test_tail_state import make_tail

if TYPE_CHECKING:
    import torch


class MainDraftTestResourceManager(AttResourceManager):
    def __init__(self, device: str | torch.device | int = "cpu", *, with_tail: bool = True, tail_pool_size: int = 4):
        super().__init__(request_capacity=3)
        self.set_att_state("main", make_state(device=device))
        draft_state = GDNState(
            replace(self.main_kda_state.config, global_v_heads=2),
            request_capacity=3, mtp_step=self.main_kda_state.mtp_step, device=device,
        )
        draft_state.initialize_page_pools(big_page_capacity=3, small_page_capacity=2, pin_memory=device != "cpu")
        self.set_att_state("draft0", draft_state)
        self.set_att_state("main", full_state(device=device))
        prepared_full_state = MLAFullAttentionState(
            self.main_full_att_fp32_state.config,
            page_size=self.main_full_att_fp32_state.page_size, device=device,
        )
        prepared_full_state.initialize_storage(self.main_full_att_fp32_state.capacity)
        self.set_att_state("draft0", prepared_full_state)
        self.att_tail_type = f"indexer_kpool_c{tail_pool_size}_tail"
        if with_tail:
            tail = make_tail(device=device)
            state = KPoolAttentionTailState(
                replace(tail.config, pool_size=tail_pool_size),
                request_capacity=tail.request_capacity, mtp_step=tail.mtp_step, device=tail.tail_buffer.device,
            )
            self.set_att_state("main", state)
        self.full_att_allocator = FullAttKVAllocator(12, page_size=4, pin_memory=False)
        self.bounded_att_big_page_allocator = StatePageAllocator(3)
        self.bounded_att_small_page_allocator = StatePageAllocator(2)
        self.finish_state_creation()

    @lru_cache(maxsize=None)
    def get_cpu_transfer_layouts(self):
        raise NotImplementedError("This test manager uses explicitly supplied CPU layouts")

    @lru_cache(maxsize=None)
    def get_pd_full_att_transfer_layouts(self):
        raise NotImplementedError("This test manager uses explicitly supplied PD layouts")

    @lru_cache(maxsize=None)
    def get_pd_bounded_att_or_tail_transfer_layouts(self):
        raise NotImplementedError("This test manager uses explicitly supplied PD layouts")

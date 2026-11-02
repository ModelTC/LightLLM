from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import torch

from lightllm.common.attention_state.types import StateTransferLayout
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.common.attention_state.manager import AttentionStateManager


logger = init_logger(__name__)


class PDOperator:
    """封装 PD 分离所需的联合布局、页大小和读写操作。"""

    def __init__(self, owner: AttentionStateManager):
        self.owner = owner

    @lru_cache(maxsize=None)
    def get_pd_full_att_page_layouts(self) -> Mapping[str, StateTransferLayout]:
        page_tokens = get_env_start_args().pd_kv_page_size
        assert page_tokens > 0, "PD pages require a positive token capacity"
        layouts, offset = {}, 0
        for state in self.owner.get_full_attention_states():
            regions = tuple(
                replace(region, byte_start=offset + region.byte_start, byte_end=offset + region.byte_end)
                for region in state.get_pd_transfer_regions()
            )
            state_end = (max(region.byte_end for region in regions) + 7) // 8 * 8
            layouts[state.state_name] = StateTransferLayout(
                state_name=state.state_name,
                tokens_per_page=page_tokens,
                byte_start=offset,
                byte_end=state_end,
                regions=regions,
            )
            offset = state_end
        return layouts

    @lru_cache(maxsize=None)
    def get_pd_bounded_att_page_layouts(self) -> Mapping[str, StateTransferLayout]:
        layouts, offset = {}, 0
        for state in self.owner.get_bounded_attention_states():
            regions = tuple(
                replace(region, byte_start=offset + region.byte_start, byte_end=offset + region.byte_end)
                for region in state.get_pd_transfer_regions()
            )
            state_end = (max(region.byte_end for region in regions) + 7) // 8 * 8
            layouts[state.state_name] = StateTransferLayout(
                state_name=state.state_name,
                tokens_per_page=None,
                byte_start=offset,
                byte_end=state_end,
                regions=regions,
            )
            offset = state_end
        return layouts

    @lru_cache(maxsize=None)
    def get_page_bytes(self) -> int:
        token_layouts = self.get_pd_full_att_page_layouts()
        runtime_layouts = self.get_pd_bounded_att_page_layouts()
        full_bytes = max((layout.byte_end for layout in token_layouts.values()), default=0)
        runtime_bytes = max((layout.byte_end for layout in runtime_layouts.values()), default=0)
        if token_layouts and full_bytes < runtime_bytes:
            logger.warning(
                f"PD KV page size ({full_bytes} bytes) is smaller than the complete attention runtime "
                f"({runtime_bytes} bytes); using {runtime_bytes} bytes per transfer page"
            )
        return max(full_bytes, runtime_bytes)

    def clear_layout_cache(self) -> None:
        self.get_pd_full_att_page_layouts.cache_clear()
        self.get_pd_bounded_att_page_layouts.cache_clear()
        self.get_page_bytes.cache_clear()

    def create_ipc_copy(self) -> AttentionStateManager:
        """保留运行态 Tensor，移除 allocator 和 bounded 页池。"""
        ipc_copy = copy(self.owner)
        ipc_copy.full_att_allocator = None
        ipc_copy.bounded_att_big_page_allocator = None
        ipc_copy.bounded_att_small_page_allocator = None
        for state in self.owner.get_bounded_attention_states():
            state_copy = copy(state)
            state_copy.big_pages_pool = None
            state_copy.small_pages_pool = None
            setattr(ipc_copy, state.state_name, state_copy)
        ipc_copy._reset_operators()
        return ipc_copy

    def write_page(
        self,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        kind: Literal["kv", "att_state"],
        layouts: Mapping[str, StateTransferLayout],
        token_indices: torch.Tensor | None = None,
        req_idx: int | None = None,
        state_token_len: int | None = None,
    ) -> None:
        match kind:
            case "kv":
                assert token_indices is not None, "KV PD pages require token indices"
                for state in self.owner.get_full_attention_states():
                    state.write_to_pd_page(
                        token_indices=token_indices,
                        pd_page_buffers=pd_page_buffers,
                        pd_page_index=pd_page_index,
                        layout=layouts[state.state_name],
                    )
            case "att_state":
                assert req_idx is not None, "state PD pages require a request index"
                assert (
                    state_token_len is not None and state_token_len >= 0
                ), "PD state requires its prefill prefix length"
                for state in self.owner.get_bounded_attention_states():
                    state.write_to_pd_page(
                        req_idx=req_idx,
                        pd_page_buffers=pd_page_buffers,
                        pd_page_index=pd_page_index,
                        state_token_len=state_token_len,
                        layout=layouts[state.state_name],
                    )
            case _:
                raise AssertionError(f"unknown PD page kind {kind}")

    def read_page(
        self,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        kind: Literal["kv", "att_state"],
        layouts: Mapping[str, StateTransferLayout],
        token_indices: torch.Tensor | None = None,
        req_idx: int | None = None,
        state_token_len: int | None = None,
    ) -> None:
        match kind:
            case "kv":
                assert token_indices is not None, "KV PD pages require token indices"
                for state in self.owner.get_full_attention_states():
                    state.read_from_pd_page(
                        pd_page_buffers=pd_page_buffers,
                        token_indices=token_indices,
                        pd_page_index=pd_page_index,
                        layout=layouts[state.state_name],
                    )
            case "att_state":
                assert req_idx is not None, "state PD pages require a request index"
                assert (
                    state_token_len is not None and state_token_len >= 0
                ), "PD state requires its prefill prefix length"
                for state in self.owner.get_bounded_attention_states():
                    state.read_from_pd_page(
                        pd_page_buffers=pd_page_buffers,
                        req_idx=req_idx,
                        pd_page_index=pd_page_index,
                        state_token_len=state_token_len,
                        layout=layouts[state.state_name],
                    )
            case _:
                raise AssertionError(f"unknown PD page kind {kind}")

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING, List

import torch

from lightllm.common.attention_state.types import StateTransferLayout
from lightllm.utils.envs_utils import get_env_start_args

if TYPE_CHECKING:
    from lightllm.common.attention_state.manager import AttentionStateManager


class CpuCacheOperator:
    """实现 CPU cache 的联合布局和 State 搬运。

    AttentionStateManager 通过该 Operator 提供 CPU 页操作；这里集中处理 CPU 页如何组织，
    以及 full-attention token 与 bounded 运行态快照如何写回/读回。
    """

    def __init__(self, owner: AttentionStateManager):
        self.owner = owner

    @lru_cache(maxsize=None)
    def get_page_layouts(self) -> Mapping[str, StateTransferLayout]:
        page_tokens = get_env_start_args().cpu_cache_token_page_size
        assert page_tokens > 0, "CPU pages require a positive token capacity"
        states = (*self.owner.get_full_attention_states(), *self.owner.get_bounded_attention_states())
        assert states, "a CPU page requires participating storage"

        layouts, offset = {}, 0
        for state in states:
            regions = tuple(
                replace(region, byte_start=offset + region.byte_start, byte_end=offset + region.byte_end)
                for region in state.get_cpu_transfer_regions()
            )
            # 没有跨步历史的 State 不占 CPU 页；其余 State 仍按联合页顺序排列。
            if not regions:
                continue
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
    def get_page_bytes(self) -> int:
        layouts = self.get_page_layouts()
        page_bytes = max((layout.byte_end for layout in layouts.values()), default=0)
        return (page_bytes + 15) // 16 * 16

    def clear_layout_cache(self) -> None:
        self.get_page_layouts.cache_clear()
        self.get_page_bytes.cache_clear()

    def write_req(
        self,
        token_indices: torch.Tensor,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        layouts: Mapping[str, StateTransferLayout],
        cpu_page_ready_flags: List[bool] | None = None,
        big_page_slot_ids: List[int] | None = None,
        small_page_slot_ids: List[int] | None = None,
    ) -> None:
        """写出单个请求的 full-attention token 和 bounded 页快照。"""
        if cpu_page_ready_flags is not None:
            assert len(cpu_page_ready_flags) == len(cpu_page_indices), "CPU page and ready counts differ"
            if all(cpu_page_ready_flags):
                return

        full_states = self.owner.get_full_attention_states()
        bounded_states = self.owner.get_bounded_attention_states()
        if bounded_states:
            big_page_slot_ids = [] if big_page_slot_ids is None else big_page_slot_ids
            small_page_slot_ids = [] if small_page_slot_ids is None else small_page_slot_ids
            big_page_count = len(big_page_slot_ids)
            assert big_page_count + len(small_page_slot_ids) == len(cpu_page_indices), (
                "one state slot is required per CPU page"
            )
            big_cpu_page_indices = cpu_page_indices[:big_page_count]
            small_cpu_page_indices = cpu_page_indices[big_page_count:]
            big_page_readies = (
                cpu_page_ready_flags[:big_page_count] if cpu_page_ready_flags is not None else None
            )
            small_page_readies = (
                cpu_page_ready_flags[big_page_count:] if cpu_page_ready_flags is not None else None
            )

        for state in full_states:
            state.write_to_cpu(
                token_indices=token_indices,
                cpu_page_buffers=cpu_page_buffers,
                cpu_page_indices=cpu_page_indices,
                layout=layouts[state.state_name],
                cpu_page_ready_flags=cpu_page_ready_flags,
            )

        for state in bounded_states:
            layout = layouts.get(state.state_name)
            if layout is None:
                # 某些压缩窗口在合法缓存边界不需要快照，因此可以不进入 CPU cache。
                assert not state.get_cpu_transfer_regions(), f"missing CPU layout for {state.state_name}"
                continue
            if big_page_slot_ids:
                state.write_state_pages_to_cpu(
                    page_kind="big",
                    page_slot_ids=big_page_slot_ids,
                    cpu_page_buffers=cpu_page_buffers,
                    cpu_page_indices=big_cpu_page_indices,
                    layout=layout,
                    cpu_page_ready_flags=big_page_readies,
                )
            if small_page_slot_ids:
                state.write_state_pages_to_cpu(
                    page_kind="small",
                    page_slot_ids=small_page_slot_ids,
                    cpu_page_buffers=cpu_page_buffers,
                    cpu_page_indices=small_cpu_page_indices,
                    layout=layout,
                    cpu_page_ready_flags=small_page_readies,
                )

    def read_req(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        token_indices: torch.Tensor,
        layouts: Mapping[str, StateTransferLayout],
        big_page_slot_ids: List[int] | None = None,
        req_idx: int | None = None,
        state_token_len: int | None = None,
    ) -> None:
        """读回请求页面，并从末页恢复 bounded 运行态。"""
        full_states = self.owner.get_full_attention_states()
        bounded_states = self.owner.get_bounded_attention_states()
        if bounded_states:
            assert req_idx is not None and cpu_page_indices, "state recovery requires a target request and prefix pages"
            assert state_token_len is not None, "bounded state recovery requires its absolute prefix length"
            page_tokens = get_env_start_args().cpu_cache_token_page_size
            has_partial_page = token_indices.numel() % page_tokens != 0
            assert isinstance(big_page_slot_ids, list), "slot indices must be a list"
            assert len(big_page_slot_ids) + int(has_partial_page) == len(cpu_page_indices), (
                "slots must cover complete CPU pages only"
            )
            load_big_page_slot_ids = big_page_slot_ids.copy()
            if has_partial_page:
                load_big_page_slot_ids.append(self.owner.bounded_att_cpu_load_big_page_slot)

        for state in full_states:
            state.read_from_cpu(
                cpu_page_buffers=cpu_page_buffers,
                cpu_page_indices=cpu_page_indices,
                token_indices=token_indices,
                layout=layouts[state.state_name],
            )

        for state in bounded_states:
            if state.state_name not in layouts:
                assert not state.get_cpu_transfer_regions(), f"missing CPU layout for {state.state_name}"
                continue
            state.read_state_pages_from_cpu(
                cpu_page_buffers=cpu_page_buffers,
                cpu_page_indices=cpu_page_indices,
                big_page_slot_ids=load_big_page_slot_ids,
                layout=layouts[state.state_name],
            )

        if bounded_states:
            self.owner.restore_bounded_att_runtime_from_pages(
                req_indices=[req_idx],
                state_token_lens=[state_token_len],
                page_kind="big",
                page_slot_ids=[load_big_page_slot_ids[-1]],
            )

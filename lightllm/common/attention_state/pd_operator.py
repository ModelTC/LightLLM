from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from math import prod
from lightllm.common.attention_state.utils import weak_cached_method
from typing import TYPE_CHECKING, Literal

import torch

from lightllm.common.attention_state.types import AxisMapping, StateTransferLayout
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.common.attention_state.manager import AttentionStateManager


logger = init_logger(__name__)


def _page_layouts(states, page_tokens):
    from .manifest import digest, logical_state_groups, state_semantics

    groups = {}
    for state in states:
        regions = state.get_pd_transfer_regions()
        fields, previous_end, seen = [], 0, {}
        for region in regions:
            if region.field in seen:
                previous = seen[region.field]
                if any(
                    getattr(region, attr) != getattr(previous, attr)
                    for attr in ("shape", "axes", "dtype", "byte_start", "byte_end")
                ):
                    raise ValueError("one wire field must describe one physical byte region")
                continue
            seen[region.field] = region
            shape = list(region.shape)
            shape[region.axes.index("layer")] = 1
            alignment = region.dtype.itemsize
            padding = region.byte_start - (previous_end + alignment - 1) // alignment * alignment
            if padding < 0:
                raise ValueError("PD wire fields must be ordered and non-overlapping")
            fields.append(dict(field=region.field, shape=shape, axes=region.axes, dtype=region.dtype, padding=padding))
            previous_end = region.byte_end
        if not fields:
            continue
        key = digest(dict(role=state.model_role, **state_semantics(state), fields=fields))
        groups.setdefault(key, []).append((state, regions))

    layouts, offset = {}, 0
    ordered = sorted(
        groups.values(),
        key=lambda entries: (
            entries[0][0].model_role,
            min(group[0] for state, _ in entries for group in logical_state_groups(state)),
            entries[0][0].get_state_type(),
        ),
    )
    for entries in ordered:
        resources = sorted(group[0] for state, _ in entries for group in logical_state_groups(state))
        if len(resources) != len(set(resources)):
            raise ValueError("ambiguous logical resource in PD wire layout")
        rows = {resource: row for row, resource in enumerate(resources)}
        start, wire_regions, previous_end = offset, {}, 0
        for region in entries[0][1]:
            if region.field in wire_regions:
                continue
            shape = list(region.shape)
            shape[region.axes.index("layer")] = len(rows)
            alignment = region.dtype.itemsize
            padding = region.byte_start - (previous_end + alignment - 1) // alignment * alignment
            offset = (offset + alignment - 1) // alignment * alignment + padding
            end = offset + prod(shape) * region.dtype.itemsize
            wire_regions[region.field] = replace(region, shape=tuple(shape), byte_start=offset, byte_end=end)
            offset, previous_end = end, region.byte_end
        offset = (offset + 7) // 8 * 8
        for state, regions in entries:
            runs = []
            for local_row, group in enumerate(logical_state_groups(state)):
                external_row = rows[group[0]]
                if runs and external_row == runs[-1][1] + runs[-1][2]:
                    runs[-1][2] += 1
                else:
                    runs.append([local_row, external_row, 1])
            mapped = []
            for region in regions:
                wire = wire_regions[region.field]
                for local_row, external_row, length in runs:
                    mappings = region.axis_mappings
                    if (local_row, external_row, length) != (0, 0, len(rows)):
                        mappings = (*mappings, AxisMapping("layer", local_row, external_row, length))
                    mapped.append(replace(wire, axis_mappings=mappings, write_enabled=region.write_enabled))
            layouts[state.state_name] = StateTransferLayout(state.state_name, page_tokens, start, offset, tuple(mapped))
    return layouts


class PDOperator:
    """封装 PD 分离所需的联合布局、页大小和读写操作。"""

    def __init__(self, owner: AttentionStateManager):
        self.owner = owner

    @weak_cached_method
    def get_pd_full_att_page_layouts(self) -> Mapping[str, StateTransferLayout]:
        page_tokens = get_env_start_args().pd_kv_page_size
        assert page_tokens > 0, "PD pages require a positive token capacity"
        return _page_layouts(self.owner.get_full_attention_states(), page_tokens)

    @weak_cached_method
    def get_pd_bounded_att_page_layouts(self) -> Mapping[str, StateTransferLayout]:
        return _page_layouts(self.owner.get_bounded_attention_states(), None)

    @weak_cached_method
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

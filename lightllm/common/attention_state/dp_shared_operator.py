from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, List

import torch

from .token_copy import TokenPageCopyPlan, TransferBatch
from .types import ResourceId
from .bounded_att.peer_copy import BoundedPeerCopy

if TYPE_CHECKING:
    from lightllm.common.attention_state.manager import AttentionStateManager


@dataclass(frozen=True)
class _PeerCopyPlan:
    token_plans: tuple
    legacy_tokens: tuple
    bounded_plans: tuple


def _resources(manager):
    resources = {}
    for state in manager.get_full_attention_states():
        for row, group in enumerate(state.config.logical_layer_groups):
            resource = ResourceId(state.model_role, group[0])
            if resource in resources:
                raise ValueError(f"ambiguous DP resource {resource}")
            resources[resource] = state, row, frozenset(group)
    return resources


def _bounded_resources(manager):
    resources = {}
    for state in manager.get_bounded_attention_states():
        for row, layer in enumerate(state.config.logical_layer_ids):
            resource = ResourceId(state.model_role, layer, state.resource_kind)
            if resource in resources:
                raise ValueError(f"ambiguous DP bounded resource {resource}")
            resources[resource] = state, row
    return resources


def _representation(state):
    def without_rows(value):
        if isinstance(value, dict):
            return {key: without_rows(item) for key, item in value.items() if key != "logical_layer_groups"}
        return value

    format = state.get_attention_format()
    representation = asdict(format) if format is not None else None
    if representation is not None:
        representation.pop("page_size", None)
    return without_rows(asdict(state.config)), representation


def _layer_fields(fields, start, length):
    return {
        name: replace(field, tensor=field.tensor.narrow(field.axes.index("layer"), start, length))
        for name, field in fields.items()
    }


class DPSharedOperator:
    """封装 DP shared 的跨副本复制、IPC 导出和共享页注册。"""

    def __init__(self, owner: AttentionStateManager):
        self.owner = owner
        self._peer_plans = {}

    def bind_peer(self, remote_att_manager: AttentionStateManager):

        if remote_att_manager in self._peer_plans:
            return self._peer_plans[remote_att_manager]
        if self.owner.model_identities != remote_att_manager.model_identities:
            raise ValueError("DP replicas have different model or weight identities")
        local_resources, remote_resources = _resources(self.owner), _resources(remote_att_manager)
        if local_resources.keys() != remote_resources.keys():
            raise ValueError("DP replicas have different logical resources")
        plans, legacy, descriptions, static_scales = [], [], {}, {}

        def describe(state):
            if state not in descriptions:
                descriptions[state] = _representation(state)
            return descriptions[state]

        def scales(state):
            if state not in static_scales:
                value = getattr(state, "scales", None)
                static_scales[state] = None if value is None else value.detach().cpu()
            return static_scales[state]

        for state in self.owner.get_full_attention_states():
            fields = state.get_token_copy_fields()
            if fields is None:
                remote = getattr(remote_att_manager, state.state_name, None)
                if remote is None or state.config.logical_layer_groups != remote.config.logical_layer_groups:
                    raise ValueError("this DP representation requires identical physical layer groups")
                legacy.append((state, remote))
                continue
            runs = []
            for row, group in enumerate(state.config.logical_layer_groups):
                resource = ResourceId(state.model_role, group[0])
                remote, remote_row, consumers = remote_resources[resource]
                if consumers != frozenset(group) or describe(state) != describe(remote):
                    raise ValueError(f"incompatible DP sharing, representation or shard: {resource}")
                local_scale, remote_scale = scales(state), scales(remote)
                if (local_scale is None) != (remote_scale is None) or (
                    local_scale is not None and not torch.equal(local_scale[row], remote_scale[remote_row])
                ):
                    raise ValueError(f"incompatible DP KV calibration: {resource}")
                if runs and remote is runs[-1][0] and remote_row == runs[-1][1] + runs[-1][3]:
                    runs[-1][3] += 1
                else:
                    runs.append([remote, remote_row, row, 1])
            for remote, remote_row, row, count in runs:
                remote_fields = remote.get_token_copy_fields()
                if remote_fields is None:
                    raise ValueError("DP native copy requires native fields at both endpoints")
                plans.append(
                    TokenPageCopyPlan.for_tokens(
                        _layer_fields(remote_fields, remote_row, count), _layer_fields(fields, row, count)
                    )
                )
        local_bounded, remote_bounded = _bounded_resources(self.owner), _bounded_resources(remote_att_manager)
        if local_bounded.keys() != remote_bounded.keys():
            raise ValueError("DP replicas require the same complete bounded state set")
        bounded = []
        for state in self.owner.get_bounded_attention_states():
            runs = []
            for row, layer in enumerate(state.config.logical_layer_ids):
                remote, remote_row = remote_bounded[ResourceId(state.model_role, layer, state.resource_kind)]
                if runs and remote is runs[-1][0] and remote_row == runs[-1][1] + runs[-1][3]:
                    runs[-1][3] += 1
                else:
                    runs.append([remote, remote_row, row, 1])
            bounded.extend(BoundedPeerCopy.bind(remote, state, src, dst, count) for remote, src, dst, count in runs)
        plan = _PeerCopyPlan(tuple(plans), tuple(legacy), tuple(bounded))
        self._peer_plans[remote_att_manager] = plan
        return plan

    def create_ipc_copy(self) -> AttentionStateManager:
        """保留共享运行态和共享 CPU 页池，排除本地 allocator。"""
        for state in self.owner.get_bounded_attention_states():
            for pool in (state.big_pages_pool, state.small_pages_pool):
                if pool is not None:
                    for tensor in vars(pool).values():
                        if isinstance(tensor, torch.Tensor):
                            assert (
                                tensor.device.type == "cpu" and tensor.is_shared()
                            ), "DP IPC requires shared CPU page pools"
        ipc_copy = copy(self.owner)
        ipc_copy.full_att_allocator = None
        ipc_copy.bounded_att_big_page_allocator = None
        ipc_copy.bounded_att_small_page_allocator = None
        ipc_copy._reset_operators()
        return ipc_copy

    def register_page_pools_pin_memory(self) -> None:
        """在当前 GPU 映射所有共享 bounded 页池。"""
        for state in self.owner.get_bounded_attention_states():
            if state.big_pages_pool is not None:
                state.big_pages_pool.register_pin_memory()
            if state.small_pages_pool is not None:
                state.small_pages_pool.register_pin_memory()

    def copy_from_dp(
        self,
        remote_att_manager: AttentionStateManager,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        src_big_page_ids: List[int],
        dst_big_page_ids: List[int],
        src_req_idx: int,
        dst_req_idx: int,
    ) -> None:
        """复制其他 DP 的 token、大页快照和已接受的运行态。"""
        assert type(remote_att_manager) is type(self.owner), "DP replicas must use the same model resource manager"
        assert src_token_indices.numel() == dst_token_indices.numel(), "source and destination token counts differ"
        assert len(src_big_page_ids) == len(dst_big_page_ids), "source and destination page counts differ"
        plan = self.bind_peer(remote_att_manager)
        if src_token_indices.numel():
            batch = TransferBatch(src_token_indices, dst_token_indices)
            for token_plan in plan.token_plans:
                token_plan.validate_batch(batch)
            for token_plan in plan.token_plans:
                token_plan.run(batch)
            for state, remote in plan.legacy_tokens:
                state.copy_tokens_from_dp(
                    src_token_indices=src_token_indices,
                    dst_token_indices=dst_token_indices,
                    remote_dp_full_att_state=remote,
                )
        if src_big_page_ids:
            for bounded in plan.bounded_plans:
                bounded.copy_pages(src_big_page_ids, dst_big_page_ids)
        for bounded in plan.bounded_plans:
            bounded.copy_runtime([src_req_idx], [dst_req_idx])

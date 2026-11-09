"""GDN replay 的 bounded attention 状态。

直接实现状态接口，不继承 native GDNState。history 在 replay_gdn_runtime，
页布局在 replay_gdn_checkpoint，已接受快照在 replay_gdn_snapshot。
一层 decode 重放在 replay_gdn_step。
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch

from ...utils import deepcopy_result, dtype_name
from ..base_state import BoundedAttentionState
from ..conv_ssm_types import ConvSSMStateConfig
from .replay_gdn_checkpoint import (
    build_transfer_regions,
    copy_big_pages,
    init_page_pools,
    read_pages_from_cpu,
    write_pages_to_cpu,
)
from .replay_gdn_runtime import ReplayGDNRuntime
from .replay_gdn_snapshot import (
    accepted_conv,
    copy_runtime_from_dp,
    read_pd,
    restore_runtime,
    save_runtime,
    write_pd,
)


class _AcceptedTensors(NamedTuple):
    conv: torch.Tensor
    ssm: torch.Tensor


def _replay_settings(config: dict | None, mtp_step: int) -> tuple[int, str]:
    config = dict(config or {})
    capacity = int(config.get("cache_len", 16))
    compute_mode = config.get("compute_mode", "precompute" if mtp_step > 0 else "inline")
    return capacity, compute_mode


class ReplayGDNState(BoundedAttentionState):
    """GDN replay 运行态。槽位布局和 history 都在本类，不沿用 native 的 MTP 槽。"""

    def __init__(
        self,
        config: ConvSSMStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
        *,
        ssm_state_mode: str = "replay",
        ssm_replay_config: dict | None = None,
        gate_lower_bound: float = -5.0,
    ) -> None:
        """创建单槽 canonical SSM，并挂上只属于当前请求的 replay history。"""
        if ssm_state_mode != "replay":
            raise ValueError(f"ReplayGDNState only implements replay mode, got {ssm_state_mode}")
        assert isinstance(config, ConvSSMStateConfig), "config must be ConvSSMStateConfig"
        self.config = config
        self.state_name = None
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self.gate_lower_bound = gate_lower_bound
        self._algorithm = "gdn"
        self._logical_to_storage_layer = {layer_id: index for index, layer_id in enumerate(config.logical_layer_ids)}
        layer_count = len(config.logical_layer_ids)
        self.conv_state = torch.zeros(
            (
                layer_count,
                request_capacity,
                config.conv_channels,
                config.conv_width + mtp_step,
            ),
            dtype=config.conv_dtype,
            device=device,
        )
        self.ssm_state = torch.zeros(
            (
                layer_count,
                request_capacity,
                config.num_v_heads,
                config.key_dim,
                config.value_dim,
            ),
            dtype=config.ssm_dtype,
            device=device,
        )
        self.mtp_state_index = torch.zeros(request_capacity, dtype=torch.int32, device=device) if mtp_step else None
        self.big_pages_pool = None
        self.small_pages_pool = None
        capacity, compute_mode = _replay_settings(ssm_replay_config, mtp_step)
        self.runtime = ReplayGDNRuntime(
            self.ssm_state,
            capacity,
            mtp_step + 1,
            config.conv_dtype,
            num_key_heads=config.num_k_heads,
            kda=False,
            gate_lower_bound=gate_lower_bound,
            compute_mode=compute_mode,
        )

    @property
    def ssm_slots_per_req(self) -> int:
        """每个请求只保留一份 canonical SSM。"""

        return 1

    @property
    def _ssm_executor(self):
        """测试沿用的 history 入口。"""

        return self.runtime

    def get_state_type(self) -> str:
        """返回算法名和 Conv/SSM 存储精度。"""
        return (
            f"{self._algorithm}_conv_{dtype_name(dtype=self.config.conv_dtype)}"
            f"_ssm_{dtype_name(dtype=self.config.ssm_dtype)}"
        )

    def initialize_page_pools(
        self,
        big_page_capacity: int,
        small_page_capacity: int = 0,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        init_page_pools(self, big_page_capacity, small_page_capacity, device, pin_memory)

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        return logical_layer_id in self._logical_to_storage_layer

    def storage_layer(self, logical_layer_id: int) -> int:
        """返回逻辑层在 history 和 canonical state 里的存储下标。"""

        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int):
        index = self.storage_layer(logical_layer_id)
        return self.conv_state[index], self.ssm_state[index]

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self):
        return build_transfer_regions(self)

    def write_state_pages_to_cpu(
        self,
        page_kind,
        page_slot_ids,
        cpu_page_buffers,
        cpu_page_indices,
        layout,
        cpu_page_ready_flags=None,
    ):
        write_pages_to_cpu(
            self,
            page_kind,
            page_slot_ids,
            cpu_page_buffers,
            cpu_page_indices,
            layout,
            cpu_page_ready_flags,
        )

    def read_state_pages_from_cpu(self, cpu_page_buffers, cpu_page_indices, big_page_slot_ids, layout):
        read_pages_from_cpu(self, cpu_page_buffers, cpu_page_indices, big_page_slot_ids, layout)

    def copy_big_pages_from_dp(self, src_page_ids, dst_page_ids, remote_dp_bounded_state):
        copy_big_pages(self, src_page_ids, dst_page_ids, remote_dp_bounded_state)

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self):
        return build_transfer_regions(self)

    def reset_bounded_att_state(self, req_indices: list[int]) -> None:
        """清零请求的 Conv、单槽 SSM、history 和 MTP 游标。"""
        assert isinstance(req_indices, list), "request indices must be a list"
        for req_idx in req_indices:
            self.conv_state[:, req_idx].zero_()
            self.ssm_state[:, req_idx].zero_()
            self.runtime.clear_history(req_idx)
            if self.mtp_state_index is not None:
                self.mtp_state_index[req_idx].fill_(0)

    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind,
        page_slot_ids: list[int],
    ) -> None:
        save_runtime(self, req_indices, state_token_lens, page_kind, page_slot_ids)

    def restore_bounded_att_runtime_from_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind,
        page_slot_ids: list[int],
    ) -> None:
        restore_runtime(self, req_indices, state_token_lens, page_kind, page_slot_ids)

    def commit_bounded_att_state(self, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """提交已接受前缀，并推进 replay cursor。"""
        if self.mtp_state_index is None:
            return
        self.runtime.commit_accepted(self.mtp_state_index, req_indices, accepted_steps)

    def merge_accepted_updates(self, reqs: torch.Tensor) -> None:
        """Prefill 前把已接受 history 折回 canonical state。"""
        self.runtime.merge_accepted_updates(reqs)

    def get_accepted_state(self, req_idx: int) -> _AcceptedTensors:
        """返回已接受 Conv 窗口和物化 SSM，不改写 canonical state。"""
        return _AcceptedTensors(accepted_conv(self, req_idx), self.runtime.get_accepted_state(req_idx))

    def copy_runtime_state_from_dp(
        self,
        src_req_indices: list[int],
        dst_req_indices: list[int],
        remote_dp_bounded_state: "ReplayGDNState",
    ) -> None:
        copy_runtime_from_dp(self, src_req_indices, dst_req_indices, remote_dp_bounded_state)

    def write_to_pd_page(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout,
    ) -> None:
        write_pd(self, req_idx, pd_page_buffers, pd_page_index, state_token_len, layout)

    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout,
    ) -> None:
        read_pd(self, pd_page_buffers, req_idx, pd_page_index, state_token_len, layout)

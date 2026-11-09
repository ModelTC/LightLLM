"""KDA replay 的 bounded attention 状态。

Conv、canonical SSM、replay history、页布局和已接受快照都在本类。
一层 decode 重放在 replay_kda_decode。
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from math import prod
from typing import NamedTuple

import torch
import triton

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import (
    _accept_updates,
    _fold_history,
    _materialize,
    _prepare_fold,
)
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_acceptance import commit_ssm_prefix
from lightllm.utils.log_utils import init_logger

from ...types import AxisMapping, PageKind, StateTransferLayout, TensorField, TransferRegion
from ...utils import deepcopy_result, dtype_name
from ..base_state import BoundedAttentionState
from ..conv_ssm_types import ConvSSMStateConfig
from ..copy_kernel import copy_bounded_att_tensors
from ..transfer_utils import bounded_att_buffer_copies
from .kda_pages_pool import KDAPagesPool

logger = init_logger(__name__)


class ReplayKDAState(BoundedAttentionState):
    """KDA replay 运行态。Conv、单槽 SSM 和 history 都在本类，不沿用 native 的 MTP 槽。"""

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        config: ConvSSMStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
        ssm_state_mode: str = "replay",
        ssm_replay_config: dict | None = None,
        gate_lower_bound: float = -5.0,
    ) -> None:
        """创建单槽 canonical SSM，并挂上只属于当前请求的 replay history。"""
        if ssm_state_mode != "replay":
            raise ValueError(f"ReplayKDAState only implements replay mode, got {ssm_state_mode}")
        assert isinstance(config, ConvSSMStateConfig), "config must be ConvSSMStateConfig"
        self.config = config
        self.state_name = None
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self.gate_lower_bound = gate_lower_bound
        self._algorithm = "kda"
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
        _bind_history(
            self,
            self.ssm_state,
            capacity,
            mtp_step + 1,
            config.conv_dtype,
            config.num_k_heads,
            gate_lower_bound,
            compute_mode,
            None,
        )

    @classmethod
    def from_ssm(
        cls,
        ssm: torch.Tensor,
        capacity: int,
        verify_width: int,
        activation_dtype: torch.dtype = torch.bfloat16,
        num_key_heads: int | None = None,
        kda: bool = True,
        gate_lower_bound: float = -5.0,
        compute_mode: str = "inline",
        run_config: dict[str, int] | None = None,
    ) -> "ReplayKDAState":
        """用已有 SSM 张量组装 history。kernel 测试和 autotune scratch 走这里。"""

        if not kda:
            raise ValueError("KDA replay requires the KDA gate")
        owner = cls.__new__(cls)
        owner.config = None
        owner.state_name = None
        owner.request_capacity = ssm.shape[1]
        owner.mtp_step = verify_width - 1
        owner._algorithm = "kda"
        owner._logical_to_storage_layer = {}
        owner.conv_state = None
        owner.mtp_state_index = None
        owner.big_pages_pool = None
        owner.small_pages_pool = None
        _bind_history(
            owner,
            ssm,
            capacity,
            verify_width,
            activation_dtype,
            num_key_heads,
            gate_lower_bound,
            compute_mode,
            run_config,
        )
        return owner

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
        """容量规划完成后创建本算法的大小页池；容量为零的页池保持 None。"""
        assert self.big_pages_pool is None and self.small_pages_pool is None, "page pools already initialized"
        assert big_page_capacity >= 0 and small_page_capacity >= 0, "page capacities must be nonnegative"
        self.big_pages_pool = (
            KDAPagesPool(
                size=big_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if big_page_capacity
            else None
        )
        self.small_pages_pool = (
            KDAPagesPool(
                size=small_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if small_page_capacity
            else None
        )

    # ========================================================================
    # 二、计算视图、运行态重置、已接受快照与 MTP 提交
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射。"""
        return logical_layer_id in self._logical_to_storage_layer

    def storage_layer(self, logical_layer_id: int) -> int:
        """返回逻辑层在 history 和 canonical state 里的存储下标。"""

        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int):
        index = self.storage_layer(logical_layer_id)
        return self.conv_state[index], self.ssm_state[index]

    def reset_bounded_att_state(self, req_indices: list[int]) -> None:
        """清零请求的 Conv、单槽 SSM、history 和 MTP 游标。"""
        assert isinstance(req_indices, list), "request indices must be a list"
        for req_idx in req_indices:
            self.conv_state[:, req_idx].zero_()
            self.ssm_state[:, req_idx].zero_()
            self.clear_history(req_idx)
            if self.mtp_state_index is not None:
                self.mtp_state_index[req_idx].fill_(0)

    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind,
        page_slot_ids: list[int],
    ) -> None:
        """把已接受 Conv 窗口和物化 SSM 写入页，不改 canonical state。"""
        assert isinstance(req_indices, list) and isinstance(page_slot_ids, list), "indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        copy_bounded_att_tensors(
            copies=[
                (self._accepted_conv(req_idx), pool.conv_state[slot_id])
                for req_idx, slot_id in zip(req_indices, page_slot_ids)
            ]
        )
        self.export_ssm(req_indices, [pool.ssm_state[slot_id] for slot_id in page_slot_ids])

    def restore_bounded_att_runtime_from_pages(
        self,
        req_indices: list[int],
        state_token_lens: list[int],
        page_kind,
        page_slot_ids: list[int],
    ) -> None:
        """从页恢复基准 Conv/SSM，并清空目标请求的 history。"""
        assert isinstance(req_indices, list) and isinstance(page_slot_ids, list), "indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        width = self.config.conv_width
        conv_copies, ssm_copies = [], []
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            self.clear_history(req_idx)
            conv_copies.append((pool.conv_state[slot_id], self.conv_state[:, req_idx, :, :width]))
            ssm_copies.append((pool.ssm_state[slot_id], self.ssm_state[:, req_idx]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)
        if self.mtp_state_index is not None:
            for req_idx in req_indices:
                self.mtp_state_index[req_idx].fill_(0)

    def commit_bounded_att_state(self, req_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
        """提交已接受前缀，折进 canonical SSM，并清空 history。"""
        if self.mtp_state_index is None:
            return
        self.commit_accepted(self.mtp_state_index, req_indices, accepted_steps)
        if self.ssm_state.is_cuda:
            self.merge_accepted_updates(req_indices)

    def get_accepted_state(self, req_idx: int) -> _AcceptedTensors:
        """返回已接受 Conv 窗口和物化 SSM，不改写 canonical state。"""
        return _AcceptedTensors(self._accepted_conv(req_idx), self.accepted_ssm(req_idx))

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回已接受 Conv/SSM 快照的字段区间。不含 history 或 MTP 候选区。"""
        return self._transfer_regions()

    def write_state_pages_to_cpu(
        self,
        page_kind: PageKind,
        page_slot_ids: list[int],
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: list[int],
        layout: StateTransferLayout,
        cpu_page_ready_flags: list[bool] | None = None,
    ) -> None:
        """将指定页池槽位的快照写入 CPU 页。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        pool = self._pool(page_kind)
        assert isinstance(page_slot_ids, list), "slot indices must be a list"
        assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
        assert len(page_slot_ids) == len(cpu_page_indices), "CPU page and slot counts differ"
        if cpu_page_ready_flags is not None:
            assert len(cpu_page_ready_flags) == len(cpu_page_indices), "CPU page and ready counts differ"
        page_fields, page_indices = [], []
        for index, (slot, page) in enumerate(zip(page_slot_ids, cpu_page_indices)):
            # ready 页不参与复制；同一传输区域的所有页面组成一个同形状批次。
            if cpu_page_ready_flags is not None and cpu_page_ready_flags[index]:
                continue
            page_fields.append(self._tensor_fields(pool.conv_state[slot], pool.ssm_state[slot]))
            page_indices.append(page)
        for copies in bounded_att_buffer_copies(
            page_fields=page_fields,
            page_buffers=cpu_page_buffers,
            layout=layout,
            page_indices=page_indices,
            write=True,
        ):
            copy_bounded_att_tensors(copies=copies)

    def read_state_pages_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: list[int],
        big_page_slot_ids: list[int],
        layout: StateTransferLayout,
    ) -> None:
        """读入指定大页槽位。请求运行态由 restore_bounded_att_runtime_from_pages 另行恢复。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        pool = self.big_pages_pool
        assert isinstance(big_page_slot_ids, list), "slot indices must be a list"
        assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
        assert len(cpu_page_indices) == len(big_page_slot_ids), "CPU page and slot counts differ"
        # 本地目标保留本 TP rank 的分片形状。layout.regions 描述 CPU 全局字段。
        pool_page_fields = [
            self._tensor_fields(pool.conv_state[slot], pool.ssm_state[slot]) for slot in big_page_slot_ids
        ]
        for copies in bounded_att_buffer_copies(
            page_fields=pool_page_fields,
            page_buffers=cpu_page_buffers,
            layout=layout,
            page_indices=cpu_page_indices,
            write=False,
        ):
            copy_bounded_att_tensors(copies=copies)

    # ========================================================================
    # 四、DP 迁移：大页快照与基准运行态
    # ========================================================================

    def copy_big_pages_from_dp(self, src_page_ids: list[int], dst_page_ids: list[int], remote_dp_bounded_state) -> None:
        """复制同一 TP 分片的大页 Conv/SSM 快照，保留源页和双方请求运行态。"""
        assert isinstance(src_page_ids, list) and isinstance(dst_page_ids, list), "page indices must be lists"
        assert len(src_page_ids) == len(dst_page_ids), "source and destination page counts differ"
        assert (
            type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config
        ), "DP copy requires the same algorithm and equivalent TP state configurations"
        src_pool = remote_dp_bounded_state.big_pages_pool
        dst_pool = self.big_pages_pool
        assert src_pool is not None and dst_pool is not None, "DP big page copy requires both page pools"
        # 大页是基准快照，与请求槽位无关。页池须在本进程注册，当前流访问完才能归还页面。
        conv_copies, ssm_copies = [], []
        for src_page, dst_page in zip(src_page_ids, dst_page_ids):
            conv_copies.append((src_pool.conv_state[src_page], dst_pool.conv_state[dst_page]))
            ssm_copies.append((src_pool.ssm_state[src_page], dst_pool.ssm_state[dst_page]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)

    def copy_runtime_state_from_dp(
        self,
        src_req_indices: list[int],
        dst_req_indices: list[int],
        remote_dp_bounded_state: "ReplayKDAState",
    ) -> None:
        """复制已接受运行态到目标单槽，并清空目标 history。"""
        assert isinstance(src_req_indices, list) and isinstance(dst_req_indices, list), "request indices must be lists"
        assert len(src_req_indices) == len(dst_req_indices), "source and destination request counts differ"
        assert (
            type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config
        ), "DP copy requires the same algorithm and equivalent TP state configurations"
        width = self.config.conv_width
        conv_copies = []
        for src_req, dst_req in zip(src_req_indices, dst_req_indices):
            self.clear_history(dst_req)
            conv_copies.append(
                (
                    remote_dp_bounded_state._accepted_conv(src_req),
                    self.conv_state[:, dst_req, :, :width],
                )
            )
        copy_bounded_att_tensors(copies=conv_copies)
        remote_dp_bounded_state.export_ssm(src_req_indices, [self.ssm_state[:, dst_req] for dst_req in dst_req_indices])
        if self.mtp_state_index is not None:
            for dst_req in dst_req_indices:
                self.mtp_state_index[dst_req].fill_(0)

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 PD 传输字段与切片，交由管理器安排其在联合页中的位置。"""
        return self._transfer_regions()

    def write_to_pd_page(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """写出已接受 Conv 窗口和物化 SSM，不改写源 history。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        accepted = self.get_accepted_state(req_idx)
        fields = self._tensor_fields(accepted.conv, accepted.ssm)
        for copies in bounded_att_buffer_copies(
            page_fields=[fields],
            page_buffers=pd_page_buffers,
            layout=layout,
            page_indices=[pd_page_index],
            write=True,
        ):
            copy_bounded_att_tensors(copies=copies)

    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """读入基准 Conv/SSM，并清空该请求的 history。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        self.clear_history(req_idx)
        fields = self._tensor_fields(
            self.conv_state[:, req_idx, :, : self.config.conv_width],
            self.ssm_state[:, req_idx],
        )
        for copies in bounded_att_buffer_copies(
            page_fields=[fields],
            page_buffers=pd_page_buffers,
            layout=layout,
            page_indices=[pd_page_index],
            write=False,
        ):
            copy_bounded_att_tensors(copies=copies)
        if self.mtp_state_index is not None:
            self.mtp_state_index[req_idx].fill_(0)

    # ========================================================================
    # 六、Replay history
    # ========================================================================
    def close_history_after_layer(self, logical_layer_id: int, reqs: torch.Tensor, fold: bool) -> None:
        """本状态最后一层结束时收口 history。fold 时先把已接受前缀折进 canonical SSM。"""

        if self.config is None or not self.config.logical_layer_ids:
            return
        if logical_layer_id != self.config.logical_layer_ids[-1]:
            return
        if fold:
            self.merge_accepted_updates(reqs)
        else:
            self.clear_history(reqs)

    def clear_history(self, req: int | torch.Tensor) -> None:
        """清空复用或恢复请求的临时 history，保留 canonical SSM。"""

        if type(req) is int:
            # 标量赋值会把 CPU tensor 复制到 CUDA slot。
            self.cursors[req].zero_()
        else:
            self.cursors[req] = 0

    def prepare_decode(self, reqs: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        """保存本轮 decode 的 history 起点，并折叠即将溢出的旧半区。"""

        positions = torch.empty_like(reqs)
        layers, _, hv, kd, vd = self.ssm_state.shape
        active = torch.empty((reqs.numel() + 1,), dtype=torch.int32, device=reqs.device)
        _prepare_fold[(1,)](
            reqs,
            self.cursors,
            positions,
            self.forward_cursors,
            cu_seqlens,
            active,
            reqs.numel(),
            self.hold_req_id,
            self.capacity,
            cu_seqlens is not None,
            triton.next_power_of_2(reqs.numel()),
            self.verify_width,
            num_warps=4,
        )
        programs = min(self.fold_programs, triton.cdiv(vd, 32) * reqs.numel() * layers * hv)
        _fold_history[(programs,)](
            self.ssm_state,
            self.raw_keys,
            self.raw_values,
            self.gates,
            self.betas,
            reqs,
            positions,
            active,
            reqs.numel(),
            layers,
            hv,
            self.num_key_heads,
            kd,
            vd,
            self.capacity,
            triton.next_power_of_2(kd),
            32,
            self.ssm_state.stride(0),
            self.raw_keys.stride(0),
            self.raw_values.stride(0),
            self.gates.stride(0),
            self.betas.stride(0),
            True,
            num_warps=2,
            num_stages=1,
        )
        return positions

    def accept_updates(self, reqs: torch.Tensor, accepted_state_indices: torch.Tensor | None = None) -> None:
        """更新 cursor，保留每个请求本轮已接受的前缀。"""

        _accept_updates[(triton.cdiv(reqs.numel(), 256),)](
            self.cursors,
            reqs,
            accepted_state_indices,
            reqs.numel(),
            self.hold_req_id,
            self.verify_width,
            256,
        )

    def accept_single_token(self, reqs: torch.Tensor) -> None:
        """非 MTP decode 把单 token 记入当前 history。"""

        self.accept_updates(reqs)

    def merge_accepted_updates(self, reqs: torch.Tensor) -> None:
        """把已接受 history 折回 canonical SSM，并清空 cursor。"""

        if reqs.numel() == 0:
            return
        self._materialize_accepted(reqs, self.ssm_state, snapshot=False)
        self.cursors.index_fill_(0, reqs.long(), 0)

    def accepted_ssm(self, req_idx: int) -> torch.Tensor:
        """返回不修改 canonical SSM 和 history 的单请求快照。"""

        layers, _, hv, k, v = self.ssm_state.shape
        output = torch.empty((layers, hv, k, v), dtype=self.ssm_state.dtype, device=self.ssm_state.device)
        reqs = torch.tensor([req_idx], dtype=torch.int32, device=self.ssm_state.device)
        self._materialize_accepted(reqs, output, snapshot=True)
        return output

    def export_accepted_state(
        self,
        req_indices: list[int],
        targets: list[torch.Tensor],
        accepted_indices: torch.Tensor | None = None,
        flush: bool = False,
    ) -> None:
        """将多个 accepted 快照直接写入调用方目标页。"""

        if not req_indices:
            return
        shape = (self.ssm_state.shape[0], *self.ssm_state.shape[2:])
        strides = targets[0].stride()
        assert len(req_indices) == len(targets)
        assert all(t.shape == shape and t.stride() == strides and t.dtype == self.ssm_state.dtype for t in targets)
        self._materialize_accepted(
            _upload_indices(req_indices, self.ssm_state.device, torch.int32),
            _upload_indices([t.data_ptr() for t in targets], self.ssm_state.device),
            snapshot=not flush,
            flush_export=flush,
            output_strides=strides,
        )

    def commit_accepted(self, indices: torch.Tensor, reqs: torch.Tensor, lengths: torch.Tensor) -> None:
        """把接受长度写入 MTP 槽位索引，并推进本请求的 replay cursor。"""

        if self.verify_width > 1:
            commit_ssm_prefix(indices, reqs, lengths, self.cursors)
        else:
            commit_ssm_prefix(indices, reqs, lengths)
            self.accept_updates(reqs, indices)

    def export_ssm(self, req_indices: list[int], targets: list[torch.Tensor], flush: bool = False) -> None:
        """把已接受 SSM 写入调用方页面。CUDA 目标直接写指针，其他设备先物化再拷贝。"""

        if not req_indices:
            return
        target = targets[0]
        if target.is_cuda and target.device == self.ssm_state.device:
            self.export_accepted_state(req_indices, targets, flush=flush)
            return
        for req, dst in zip(req_indices, targets):
            snap = torch.empty(
                (self.ssm_state.shape[0], *self.ssm_state.shape[2:]),
                dtype=self.ssm_state.dtype,
                device=self.ssm_state.device,
            )
            self._materialize_accepted(
                torch.tensor([req], dtype=torch.int32, device=self.ssm_state.device),
                snap,
                snapshot=True,
            )
            dst.copy_(snap)
            if flush:
                self.ssm_state[:, req].copy_(snap)
                self.clear_history(req)

    _run_config: dict[str, int] | None = None

    @property
    def run_config(self) -> dict[str, int] | None:
        """返回物化和 autotune 使用的 Triton 配置，不是 SSM 布局。"""

        return self._run_config

    @run_config.setter
    def run_config(self, config: dict | None) -> None:
        """保存一份与调用方脱开的 kernel 配置。"""

        if config is None:
            self._run_config = None
            return
        self._run_config = {
            "BV": int(config["BV"]),
            "num_warps": int(config["num_warps"]),
            "num_stages": int(config.get("num_stages", 3)),
        }

    # ========================================================================
    # 七、私有辅助实现
    # ========================================================================

    def _transfer_regions(self) -> tuple[TransferRegion, ...]:
        """依次描述全局 Conv 和 SSM 区域，并标明当前 TP rank 的读写范围。"""
        # 只传有效历史窗口，不包含 MTP 候选区和游标。
        config = self.config
        storage_layer_count = len(config.logical_layer_ids)
        conv_channels = 2 * config.global_k_heads * config.key_dim + config.global_v_heads * config.value_dim
        conv_shape = (storage_layer_count, conv_channels, config.conv_width)
        conv_byte_end = prod(conv_shape) * config.conv_dtype.itemsize
        regions = []
        logger.debug(
            "%s building transfer regions: TP=%d/%d, Conv shape=%s",
            self.state_name,
            config.tp_rank,
            config.tp_world_size,
            conv_shape,
        )

        # Conv 按 Q/K/V 拼接。三个映射共用同一块全局 Conv 区域，不能重复累计字节数。
        local_channel_start = 0
        global_channel_start = 0
        for name, local_head_count, global_head_count, head_dim in (
            ("Q", config.num_k_heads, config.global_k_heads, config.key_dim),
            ("K", config.num_k_heads, config.global_k_heads, config.key_dim),
            ("V", config.num_v_heads, config.global_v_heads, config.value_dim),
        ):
            # 每组连续 rank 保存相同 heads，仅组内首 rank 写出。
            replica_count = config.tp_world_size * local_head_count // global_head_count
            head_start = config.tp_rank // replica_count * local_head_count
            write_enabled = config.tp_rank % replica_count == 0
            local_channel_count = local_head_count * head_dim
            target_channel_start = global_channel_start + head_start * head_dim
            regions.append(
                TransferRegion(
                    field="conv",
                    byte_start=0,
                    byte_end=conv_byte_end,
                    dtype=config.conv_dtype,
                    axes=("layer", "channel", "window"),
                    shape=conv_shape,
                    axis_mappings=(
                        AxisMapping(
                            axis="channel",
                            local_start=local_channel_start,
                            external_start=target_channel_start,
                            length=local_channel_count,
                        ),
                        AxisMapping(axis="window", local_start=0, external_start=0, length=config.conv_width),
                    ),
                    write_enabled=write_enabled,
                )
            )
            logger.debug(
                "%s Conv %s: local_channel_start=%d, global_channel_start=%d, "
                "channels=%d, replicas=%d, write_enabled=%s",
                self.state_name,
                name,
                local_channel_start,
                target_channel_start,
                local_channel_count,
                replica_count,
                write_enabled,
            )
            local_channel_start += local_channel_count
            global_channel_start += global_head_count * head_dim

        # SSM 沿 V head 切分，接在 Conv 之后，并按 SSM dtype 对齐字节起点。
        replica_count = config.tp_world_size * config.num_v_heads // config.global_v_heads
        head_start = config.tp_rank // replica_count * config.num_v_heads
        write_enabled = config.tp_rank % replica_count == 0
        ssm_shape = (storage_layer_count, config.global_v_heads, config.key_dim, config.value_dim)
        element_bytes = config.ssm_dtype.itemsize
        ssm_byte_start = (conv_byte_end + element_bytes - 1) // element_bytes * element_bytes
        ssm_region = TransferRegion(
            field="ssm",
            byte_start=ssm_byte_start,
            byte_end=ssm_byte_start + prod(ssm_shape) * element_bytes,
            dtype=config.ssm_dtype,
            axes=("layer", "head", "key_dim", "value_dim"),
            shape=ssm_shape,
            axis_mappings=(
                AxisMapping(axis="head", local_start=0, external_start=head_start, length=config.num_v_heads),
                AxisMapping(axis="key_dim", local_start=0, external_start=0, length=config.key_dim),
                AxisMapping(axis="value_dim", local_start=0, external_start=0, length=config.value_dim),
            ),
            write_enabled=write_enabled,
        )
        regions.append(ssm_region)
        logger.debug(
            "%s transfer regions built: SSM head_start=%d, heads=%d, write_enabled=%s, SSM bytes=[%d, %d)",
            self.state_name,
            head_start,
            config.num_v_heads,
            write_enabled,
            ssm_region.byte_start,
            ssm_region.byte_end,
        )
        return tuple(regions)

    def _pool(self, page_kind: PageKind):
        """根据大小页类型取得已启用的页池，不访问 allocator。"""
        assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        assert pool is not None, f"{page_kind} pages are disabled for {self.state_name}"
        return pool

    @staticmethod
    def _tensor_fields(conv: torch.Tensor, ssm: torch.Tensor):
        """描述一页里的 Conv 和 SSM 字段。"""
        return {
            "conv": TensorField(tensor=conv, axes=("layer", "channel", "window")),
            "ssm": TensorField(tensor=ssm, axes=("layer", "head", "key_dim", "value_dim")),
        }

    def _accepted_conv(self, req_idx: int) -> torch.Tensor:
        """取出该请求已接受的 Conv 窗口，不包含 MTP 候选区。"""
        width = self.config.conv_width
        conv = self.conv_state[:, req_idx]
        if self.mtp_state_index is None:
            return conv[:, :, :width]
        offset = self.mtp_state_index[req_idx]
        return conv.index_select(-1, torch.arange(width, device=conv.device) + offset)

    def _materialize_accepted(
        self,
        reqs: torch.Tensor,
        output: torch.Tensor,
        snapshot: bool,
        output_strides=None,
        flush_export: bool = False,
    ) -> None:
        """将请求的 accepted prefix 写入 canonical SSM 或调用方提供的输出页。"""

        assert not (snapshot and flush_export), "flush export must update the canonical state"
        layers, slots, hv, k, v = self.ssm_state.shape
        config = self.run_config or {"BV": 32, "num_warps": 4, "num_stages": 3}
        bv = config["BV"]
        _materialize[(triton.cdiv(v, bv), reqs.numel(), layers * hv)](
            self.ssm_state,
            output,
            self.raw_keys,
            self.raw_values,
            self.gates,
            self.betas,
            self.cursors,
            reqs,
            slots,
            hv,
            self.num_key_heads,
            k,
            v,
            self.capacity,
            self.hold_req_id,
            triton.next_power_of_2(k),
            bv,
            snapshot,
            True,
            FLUSH_EXPORT=flush_export,
            OUTPUT_STRIDES=output_strides,
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )


def _replay_settings(config: dict | None, mtp_step: int) -> tuple[int, str]:
    config = dict(config or {})
    capacity = int(config.get("cache_len", 16))
    compute_mode = config.get("compute_mode", "precompute" if mtp_step > 0 else "inline")
    return capacity, compute_mode


def _upload_indices(values: Sequence[int], device, dtype=torch.int64) -> torch.Tensor:
    """把 Python 索引上传到目标 device。"""

    return torch.tensor(list(values), dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def _bind_history(
    owner,
    ssm: torch.Tensor,
    capacity: int,
    verify_width: int,
    activation_dtype: torch.dtype,
    num_key_heads: int | None,
    gate_lower_bound: float,
    compute_mode: str,
    run_config: dict[str, int] | None,
) -> None:
    """在 owner 上创建 KDA replay 的双半区 history 和 cursor。"""

    assert ssm.dtype in (torch.float32, torch.bfloat16), "KDA replay requires FP32 or BF16 SSM state"
    assert verify_width > 0 and capacity >= max(4, verify_width) and capacity & (capacity - 1) == 0
    assert compute_mode in ("inline", "precompute")
    owner.ssm_state = ssm
    owner.kda = True
    owner.gate_lower_bound = gate_lower_bound
    owner.fold_programs = 8 * torch.cuda.get_device_properties(ssm.device).multi_processor_count if ssm.is_cuda else 0
    owner.capacity = capacity
    owner.verify_width = verify_width
    owner.compute_mode = compute_mode
    owner.run_config = run_config
    owner._config_is_fixed = run_config is not None
    owner.batch_configs = {}
    layers, slots, hv, k, v = ssm.shape
    owner.num_key_heads = hv if num_key_heads is None else num_key_heads
    assert owner.num_key_heads > 0 and hv % owner.num_key_heads == 0
    owner.hold_req_id = slots - 1
    owner.cursors = torch.zeros(slots, dtype=torch.int32, device=ssm.device)
    owner.forward_cursors = torch.zeros_like(owner.cursors)
    # fold 时交替使用两个半区。precompute 且 BF16、短 history 时 history 用激活精度，其余用 FP32。
    history_dtype = (
        activation_dtype
        if activation_dtype == torch.bfloat16 and verify_width > 1 and compute_mode == "precompute" and capacity <= 8
        else torch.float32
    )
    key_shape = (layers, slots, owner.num_key_heads, 2 * capacity, k)
    owner.keys = torch.empty(key_shape, dtype=history_dtype, device=ssm.device)
    owner.deltas = torch.empty((layers, slots, hv, 2 * capacity, v), dtype=history_dtype, device=ssm.device)
    owner.gates = torch.empty((layers, slots, hv, 2 * capacity, k), dtype=torch.float32, device=ssm.device)
    owner.raw_keys = torch.empty(key_shape, dtype=activation_dtype, device=ssm.device)
    owner.raw_values = torch.empty((layers, slots, hv, 2 * capacity, v), dtype=activation_dtype, device=ssm.device)
    owner.betas = torch.empty((layers, slots, hv, 2 * capacity), dtype=torch.float32, device=ssm.device)


class _AcceptedTensors(NamedTuple):
    conv: torch.Tensor
    ssm: torch.Tensor

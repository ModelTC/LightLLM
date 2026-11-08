from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List, NamedTuple
from math import prod

import torch

from lightllm.utils.log_utils import init_logger

from ..commit import StateAcceptance
from .ssm import create_ssm_executor
from .ssm.forward import SSMForward
from .ssm.export import export_indexed_state

from ..utils import deepcopy_result, dtype_name
from ..types import AxisMapping, StateTransferLayout, TensorField, TransferRegion
from .base_state import BoundedAttentionState
from .conv_ssm_types import ConvSSMStateConfig
from .transfer_utils import bounded_att_buffer_copies
from .copy_kernel import copy_bounded_att_tensors

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import PageKind


logger = init_logger(__name__)


class ConvSSMTensors(NamedTuple):
    conv: torch.Tensor
    ssm: torch.Tensor


class _ConvSSMState(BoundedAttentionState):

    algorithm: str
    pages_pool_class: type
    state_mode: str | None = None

    def __init__(
        self,
        config: ConvSSMStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
        *,
        ssm_state_mode: str = "native",
        ssm_replay_config: dict | None = None,
        gate_lower_bound: float = -5.0,
    ):
        self.config = config
        if self.state_mode is not None and ssm_state_mode != self.state_mode:
            raise ValueError(f"state mode {ssm_state_mode!r} does not match {self.state_mode!r}")
        ssm_state_mode = self.state_mode or ssm_state_mode
        self.state_name = None
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self._logical_to_storage_layer = {layer: index for index, layer in enumerate(config.logical_layer_ids)}
        self.mtp_state_index = torch.zeros(request_capacity, dtype=torch.int32, device=device) if mtp_step else None
        self.conv_state = torch.zeros(
            (len(config.logical_layer_ids), request_capacity, config.conv_channels, config.conv_width + mtp_step),
            dtype=config.conv_dtype,
            device=device,
        )
        self._ssm = create_ssm_executor(
            config,
            request_capacity,
            mtp_step,
            device,
            ssm_state_mode,
            ssm_replay_config,
            kda=self.algorithm == "kda",
            gate_lower_bound=gate_lower_bound,
        )
        self.big_pages_pool = None
        self.small_pages_pool = None

    @property
    def ssm_state(self):
        return self._ssm.state

    @property
    def ssm_slots_per_req(self):
        return self._ssm.slots_per_request

    def _reset_tracking(self, req_idx):
        self._ssm.clear_history(req_idx)
        if self.mtp_state_index is not None:
            self.mtp_state_index[req_idx].zero_()

    def reset(self, req_idx):
        self._reset_tracking(req_idx)
        self.conv_state[:, req_idx].zero_()
        start = req_idx * self.ssm_slots_per_req
        self.ssm_state[:, start : start + self.ssm_slots_per_req].zero_()

    def prepare_prefill(self, reqs):
        self._ssm.merge_accepted_updates(reqs)
        return reqs, reqs * self.ssm_slots_per_req

    def prepare_decode(self, reqs, cu_seqlens=None):
        speculative = cu_seqlens is not None
        accepted = None
        if speculative:
            if self.mtp_state_index is None:
                raise ValueError("speculative decode requires MTP state slots")
            accepted = self.mtp_state_index[reqs] + 1
            accepted = torch.where(reqs == self.request_capacity - 1, 1, accepted)
        self._ssm.prepare_decode(reqs, cu_seqlens)
        if not speculative:
            self._ssm.accept_single_token(reqs)
        return SSMForward(self, self._ssm, reqs, self._ssm.decode_indices(reqs, speculative), cu_seqlens, accepted)

    def get_accepted_state(self, req_idx):
        """Return canonical tensors; native MTP gathers and replay materialization allocate snapshots."""
        width = self.config.conv_width
        accepted = self.mtp_state_index[req_idx] if self.mtp_state_index is not None else None
        conv = self.conv_state[:, req_idx, ..., :width]
        if accepted is not None:
            offsets = torch.arange(width, device=conv.device) + accepted
            conv = self.conv_state[:, req_idx].index_select(-1, offsets)
        return ConvSSMTensors(conv, self._ssm.get_accepted_state(req_idx, accepted))

    def export_accepted_state(self, req_indices, targets):
        """Write disjoint targets on the current stream without full-state scratch or changing source history."""
        export_indexed_state(self.conv_state, req_indices, [t.conv for t in targets], self.mtp_state_index)
        self._ssm.export_accepted_state(req_indices, [t.ssm for t in targets], self.mtp_state_index)

    def prepare_restore(self, req_idx):

        self._reset_tracking(req_idx)
        return ConvSSMTensors(
            self.conv_state[:, req_idx, ..., : self.config.conv_width],
            self.ssm_state[:, req_idx * self.ssm_slots_per_req],
        )

    def restore(self, req_idx, conv, ssm):
        target = self.prepare_restore(req_idx)
        target.conv.copy_(conv, non_blocking=True)
        target.ssm.copy_(ssm, non_blocking=True)

    def get_storage_layer_index(self, logical_layer_id):
        return self._logical_to_storage_layer[logical_layer_id]

    def get_state_type(self) -> str:
        return (
            f"{self.algorithm}_conv_{dtype_name(dtype=self.config.conv_dtype)}"
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
            self.pages_pool_class(
                size=big_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if big_page_capacity
            else None
        )
        self.small_pages_pool = (
            self.pages_pool_class(
                size=small_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if small_page_capacity
            else None
        )

    # ========================================================================
    # 二、Bounded attention：计算视图、运行态重置、大小页与 MTP 提交
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，共享层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_layer_view(self, logical_layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑层号映射到物理层，返回 Conv/SSM Tensor 引用；MTP 游标和步数直接从 State 读取。"""
        storage_layer_index = self._logical_to_storage_layer[logical_layer_id]
        return self.conv_state[storage_layer_index], self.ssm_state[storage_layer_index]

    def reset_bounded_att_state(self, req_indices: List[int]) -> None:
        """清零指定请求的 Conv 历史、全部 SSM 候选及 MTP 游标，保留页池快照。"""
        assert isinstance(req_indices, list), "request indices must be a list"
        for req_idx in req_indices:
            self.reset(req_idx)

    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
        flush_history: bool = False,
    ) -> None:
        """保存各请求指定绝对 token 长度处的基准状态；Conv/SSM 不依赖长度定位。"""
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        # Hybrid radix pages are durable checkpoints. ReplaySSM can fold the
        # accepted ring into the live canonical state and the page in one
        # kernel, so the next decode starts from an empty history. MTP keeps
        # a separate Conv candidate layout and retains the old behavior.
        replay_flush = flush_history and self.mtp_state_index is None
        targets = [ConvSSMTensors(pool.conv_state[slot], pool.ssm_state[slot]) for slot in page_slot_ids]
        export_indexed_state(self.conv_state, req_indices, [target.conv for target in targets], self.mtp_state_index)
        self._ssm.export_accepted_state(
            req_indices,
            [target.ssm for target in targets],
            self.mtp_state_index,
            flush=replay_flush,
        )

    def restore_bounded_att_runtime_from_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """恢复各请求指定绝对 token 长度处的基准状态并归零游标；Conv/SSM 不依赖长度定位。"""
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        conv_copies, ssm_copies = [], []
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            target = self.prepare_restore(req_idx)
            conv_copies.append((pool.conv_state[slot_id], target.conv))
            ssm_copies.append((pool.ssm_state[slot_id], target.ssm))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)

    def validate_bounded_att_commit(self, acceptance: StateAcceptance):
        assert (
            acceptance.req_indices.device == self.ssm_state.device
        ), "commit inputs must match the state device; CUDA states require CUDA tensors"
        if self.mtp_state_index is not None:
            validator = getattr(self._ssm, "validate_commit", None)
            return validator(acceptance, self.state_name) if validator is not None else None
        return None

    def commit_bounded_att_state(self, acceptance: StateAcceptance) -> None:
        commit_data = self.validate_bounded_att_commit(acceptance)
        if self.mtp_state_index is not None:
            self._ssm.commit_validated(acceptance, self.mtp_state_index, self.state_name, commit_data=commit_data)

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回已接受 Conv/SSM 快照的字段区间与切片，供 CPU 页布局和容量计算使用。

        region.shape 描述 CPU 页中的全局字段；axis_mappings 将本地页池的 TP 分片映射到该字段。
        起止位置相对本 State，由管理器平移到 CPU 联合页；不包含 MTP 候选区或游标。
        """
        return self._get_transfer_regions()

    def write_state_pages_to_cpu(
        self,
        page_kind: PageKind,
        page_slot_ids: List[int],
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
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
            page_fields.append(self._fields(pool.conv_state[slot], pool.ssm_state[slot]))
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
        cpu_page_indices: List[int],
        big_page_slot_ids: List[int],
        layout: StateTransferLayout,
    ) -> None:
        """读入指定大页槽位；请求运行态另行通过 restore_bounded_att_runtime_from_pages 恢复。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        pool = self.big_pages_pool
        assert isinstance(big_page_slot_ids, list), "slot indices must be a list"
        assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
        assert len(cpu_page_indices) == len(big_page_slot_ids), "CPU page and slot counts differ"
        # 本地目标：选定大页槽位的 Conv/SSM 快照，保留本 TP rank 的分片形状。
        pool_page_fields = [self._fields(pool.conv_state[slot], pool.ssm_state[slot]) for slot in big_page_slot_ids]
        # 外部来源：layout.regions 描述 CPU 全局字段和 TP 范围；write=False 将其读入本地页池。
        # 窗口保持快照顺序；请求 ring 的位置由后续 restore_bounded_att_runtime_from_pages 按长度恢复。
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

    def copy_big_pages_from_dp(
        self,
        src_page_ids: List[int],
        dst_page_ids: List[int],
        remote_dp_bounded_state: _ConvSSMState,
    ) -> None:
        """复制同一 TP 分片的大页 Conv/SSM 快照，保留源页和双方请求运行态。"""
        assert isinstance(src_page_ids, list) and isinstance(dst_page_ids, list), "page indices must be lists"
        assert len(src_page_ids) == len(dst_page_ids), "source and destination page counts differ"
        assert (
            type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config
        ), "DP copy requires the same algorithm and equivalent TP state configurations"
        src_pool = remote_dp_bounded_state.big_pages_pool
        dst_pool = self.big_pages_pool
        assert src_pool is not None and dst_pool is not None, "DP big page copy requires both page pools"
        # 大页保存的是基准快照，与请求槽位及 MTP 候选区无关；页的引用和使用期限由调用方管理。
        # 页池均须在本进程注册；GPU 在当前流直接访问两端 CPU 存储，完成后才能归还页面。
        conv_copies, ssm_copies = [], []
        for src_page, dst_page in zip(src_page_ids, dst_page_ids):
            conv_copies.append((src_pool.conv_state[src_page], dst_pool.conv_state[dst_page]))
            ssm_copies.append((src_pool.ssm_state[src_page], dst_pool.ssm_state[dst_page]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)

    def copy_runtime_state_from_dp(
        self,
        src_req_indices: List[int],
        dst_req_indices: List[int],
        remote_dp_bounded_state: _ConvSSMState,
    ) -> None:
        """复制同一 TP 分片的 prefill 基准运行态并重置目标游标，不写入页池。"""
        assert isinstance(src_req_indices, list) and isinstance(dst_req_indices, list), "request indices must be lists"
        assert len(src_req_indices) == len(dst_req_indices), "source and destination request counts differ"
        assert (
            type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config
        ), "DP copy requires the same algorithm and equivalent TP state configurations"
        remote_dp_bounded_state.export_accepted_state(
            src_req_indices, [self.prepare_restore(req) for req in dst_req_indices]
        )

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 PD 传输字段与切片，交由管理器安排其在联合页中的位置。"""
        return self._get_transfer_regions()

    def write_to_pd_page(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """写出 state_token_len 处的已接受状态，不修改源 Replay 历史。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        fields = self._fields(*self.get_accepted_state(req_idx))
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
        """恢复 state_token_len 处的基准状态并规范化控制信息；Conv/SSM 不依赖长度定位。"""
        assert layout.state_name == self.state_name, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        fields = self._fields(*self.prepare_restore(req_idx))
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
    # 六、私有辅助实现
    # ========================================================================

    def _get_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """依次描述全局 Conv 和 SSM 区域，并标明当前 TP rank 的读写范围。"""
        # 阶段一：确定全局 Conv 形状；只传有效历史窗口，不包含 MTP 候选区和游标。
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

        # 阶段二：Conv 按 Q/K/V 拼接；三个映射共用同一块全局 Conv 区域，不能重复累计字节数。
        local_channel_start = 0
        global_channel_start = 0
        for name, local_head_count, global_head_count, head_dim in (
            ("Q", config.num_k_heads, config.global_k_heads, config.key_dim),
            ("K", config.num_k_heads, config.global_k_heads, config.key_dim),
            ("V", config.num_v_heads, config.global_v_heads, config.value_dim),
        ):
            # 配置已保证完整分片或副本。每组连续 rank 保存相同 heads，仅组内首 rank 写出。
            replica_count = config.tp_world_size * local_head_count // global_head_count
            head_start = config.tp_rank // replica_count * local_head_count
            write_enabled = config.tp_rank % replica_count == 0
            local_channel_count = local_head_count * head_dim
            target_channel_start = global_channel_start + head_start * head_dim
            conv_region = TransferRegion(
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
                    AxisMapping(
                        axis="window",
                        local_start=0,
                        external_start=0,
                        length=config.conv_width,
                    ),
                ),
                write_enabled=write_enabled,
            )
            regions.append(conv_region)
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

        # 阶段三：SSM 沿 V head 切分；接在 Conv 之后，按 SSM dtype 对齐字节起点。
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
                AxisMapping(
                    axis="head",
                    local_start=0,
                    external_start=head_start,
                    length=config.num_v_heads,
                ),
                AxisMapping(
                    axis="key_dim",
                    local_start=0,
                    external_start=0,
                    length=config.key_dim,
                ),
                AxisMapping(
                    axis="value_dim",
                    local_start=0,
                    external_start=0,
                    length=config.value_dim,
                ),
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
    def _fields(conv: torch.Tensor, ssm: torch.Tensor):
        return {
            "conv": TensorField(tensor=conv, axes=("layer", "channel", "window")),
            "ssm": TensorField(tensor=ssm, axes=("layer", "head", "key_dim", "value_dim")),
        }

from __future__ import annotations

from typing import TYPE_CHECKING, List
from math import prod

import torch

from lightllm.utils.log_utils import init_logger

from ..utils import dtype_name
from ..types import AxisSlice, StateTransferLayout, TensorField, TransferRegion
from .base_state import BoundedAttentionState
from .conv_ssm_types import ConvSSMStateConfig
from .gdn_pages_pool import GDNPagesPool
from .transfer_utils import bounded_att_buffer_copies
from .copy_kernel import copy_bounded_att_tensors

if TYPE_CHECKING:
    from lightllm.common.att_resources.types import PageKind


logger = init_logger(__name__)


class GDNState(BoundedAttentionState):
    """Gated DeltaNet 的 Conv 历史、SSM 和 MTP 游标，支持不同的 K/V head 数及维度。"""

    big_pages_pool: GDNPagesPool | None
    small_pages_pool: GDNPagesPool | None

    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        config: ConvSSMStateConfig,
        request_capacity: int,
        mtp_step: int = 0,
        device: str | torch.device = "cuda",
    ) -> None:
        """按请求容量创建 Conv/SSM 运行态和 MTP 游标；大小页池延后初始化。"""
        assert isinstance(config, ConvSSMStateConfig), "config must be ConvSSMStateConfig"
        self.config = config
        self.resource_id: str | None = None  # 由 AttResourceManager.set_att_state() 绑定时设置。
        self.request_capacity = request_capacity  # 包含预留的 HOLD 请求槽位。
        self.mtp_step = mtp_step
        self._logical_to_storage_layer = {
            logical_layer_id: storage_layer_index
            for storage_layer_index, logical_layer_id in enumerate(config.logical_layer_ids)
        }
        storage_layer_count = len(config.logical_layer_ids)
        self.conv_state = torch.zeros(
            (storage_layer_count, request_capacity, config.conv_channels, config.conv_width + mtp_step),
            dtype=config.conv_dtype,
            device=device,
        )
        self.ssm_state = torch.zeros(
            (storage_layer_count, request_capacity * (mtp_step + 1), config.num_v_heads, config.key_dim, config.value_dim),
            dtype=config.ssm_dtype,
            device=device,
        )
        self.mtp_state_index = torch.zeros(request_capacity, dtype=torch.int32, device=device) if mtp_step else None
        self.big_pages_pool = None
        self.small_pages_pool = None

    def get_state_type(self) -> str:
        """返回 GDN 算法及 Conv/SSM 存储精度。"""
        return (
            f"gdn_conv_{dtype_name(dtype=self.config.conv_dtype)}"
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
            GDNPagesPool(
                size=big_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if big_page_capacity
            else None
        )
        self.small_pages_pool = (
            GDNPagesPool(
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
            self.conv_state[:, req_idx].zero_()
            start = req_idx * (self.mtp_step + 1)
            self.ssm_state[:, start : start + self.mtp_step + 1].zero_()
            if self.mtp_state_index is not None:
                self.mtp_state_index[req_idx].fill_(0)

    def save_bounded_att_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        slot_ids: List[int],
    ) -> None:
        """保存各请求指定绝对 token 长度处的基准状态；Conv/SSM 不依赖长度定位。"""
        assert isinstance(req_indices, list) and isinstance(slot_ids, list), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        # TODO：大小页仅在 prefill 阶段保存，mtp_state_index 始终为 0，可直接复制基准状态。
        # 若以后支持 decode/MTP 后保存，需要按已接受游标选择 Conv 窗口和 SSM 槽位。
        conv_copies, ssm_copies = [], []
        for req_idx, slot_id in zip(req_indices, slot_ids):
            conv_copies.append((self.conv_state[:, req_idx, :, : self.config.conv_width], pool.conv_state[slot_id]))
            ssm_copies.append((self.ssm_state[:, req_idx * (self.mtp_step + 1)], pool.ssm_state[slot_id]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)

    def restore_bounded_att_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        slot_ids: List[int],
    ) -> None:
        """恢复各请求指定绝对 token 长度处的基准状态并归零游标；Conv/SSM 不依赖长度定位。"""
        assert isinstance(req_indices, list) and isinstance(slot_ids, list), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        conv_copies, ssm_copies = [], []
        for req_idx, slot_id in zip(req_indices, slot_ids):
            conv_copies.append((pool.conv_state[slot_id], self.conv_state[:, req_idx, :, : self.config.conv_width]))
            ssm_copies.append((pool.ssm_state[slot_id], self.ssm_state[:, req_idx * (self.mtp_step + 1)]))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)
        if self.mtp_state_index is not None:
            for req_idx in req_indices:
                self.mtp_state_index[req_idx].fill_(0)

    def commit_bounded_att_state(
        self,
        req_indices: torch.Tensor,
        accepted_steps: torch.Tensor,
    ) -> None:
        """用 CUDA 请求索引和 int32 接受步数选择 SSM 候选位置；步数包含基准输入步。"""
        assert req_indices.is_cuda and accepted_steps.is_cuda, "commit inputs must be CUDA tensors"
        if self.mtp_state_index is not None:
            self.mtp_state_index[req_indices] = accepted_steps - 1

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回已接受 Conv/SSM 快照的字段区间与切片，供 CPU 页布局和容量计算使用。

        region.shape 描述 CPU 页中的全局字段；axis_slices 将本地页池的 TP 分片映射到该字段。
        起止位置相对本 State，由管理器平移到 CPU 联合页；不包含 MTP 候选区或游标。
        """
        return self._get_transfer_regions()

    def write_page_state_to_cpu(
        self,
        page_kind: PageKind,
        slot_ids: List[int],
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        layout: StateTransferLayout,
        cpu_page_readies: list[bool] | None = None,
    ) -> None:
        """将指定页池槽位的快照写入 CPU 页。"""
        assert layout.resource_id == self.resource_id, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        pool = self._pool(page_kind)
        assert isinstance(slot_ids, list), "slot indices must be a list"
        assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
        assert len(slot_ids) == len(cpu_page_indices), "CPU page and slot counts differ"
        if cpu_page_readies is not None:
            assert len(cpu_page_readies) == len(cpu_page_indices), "CPU page and ready counts differ"
        page_fields, page_indices = [], []
        for index, (slot, page) in enumerate(zip(slot_ids, cpu_page_indices)):
            # ready 页不参与复制；同一传输区域的所有页面组成一个同形状批次。
            if cpu_page_readies is not None and cpu_page_readies[index]:
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

    def read_page_state_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        big_page_slot_ids: List[int],
        layout: StateTransferLayout,
    ) -> None:
        """读入指定大页槽位；请求运行态另行通过 restore_bounded_att_pages 恢复。"""
        assert layout.resource_id == self.resource_id, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        pool = self.big_pages_pool
        assert isinstance(big_page_slot_ids, list), "slot indices must be a list"
        assert isinstance(cpu_page_indices, list), "CPU page indices must be a list"
        assert len(cpu_page_indices) == len(big_page_slot_ids), "CPU page and slot counts differ"
        # 本地目标：选定大页槽位的 Conv/SSM 快照，保留本 TP rank 的分片形状。
        pool_page_fields = [self._fields(pool.conv_state[slot], pool.ssm_state[slot]) for slot in big_page_slot_ids]
        # 外部来源：layout.regions 描述 CPU 全局字段和 TP 范围；write=False 将其读入本地页池。
        # 窗口保持快照顺序；请求 ring 的位置由后续 restore_bounded_att_pages 按长度恢复。
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

    def copy_big_page_state_from_other_dp(
        self,
        src_page_ids: List[int],
        dst_page_ids: List[int],
        remote_dp_bounded_state: GDNState,
    ) -> None:
        """复制同一 TP 分片的大页 Conv/SSM 快照，保留源页和双方请求运行态。"""
        assert isinstance(src_page_ids, list) and isinstance(dst_page_ids, list), "page indices must be lists"
        assert len(src_page_ids) == len(dst_page_ids), "source and destination page counts differ"
        assert type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config, (
            "DP copy requires the same algorithm and equivalent TP state configurations"
        )
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

    def copy_run_time_state_from_other_dp(
        self,
        src_req_indices: List[int],
        dst_req_indices: List[int],
        remote_dp_bounded_state: GDNState,
    ) -> None:
        """复制同一 TP 分片的 prefill 基准运行态并重置目标游标，不写入页池。"""
        assert isinstance(src_req_indices, list) and isinstance(dst_req_indices, list), "request indices must be lists"
        assert len(src_req_indices) == len(dst_req_indices), "source and destination request counts differ"
        assert type(remote_dp_bounded_state) is type(self) and remote_dp_bounded_state.config == self.config, (
            "DP copy requires the same algorithm and equivalent TP state configurations"
        )
        # TODO：当前 DP fetch 只复制 prefill 基准状态，源 MTP 游标为 0；
        # 若以后支持 decode/MTP 后迁移，需按已接受游标选择源 Conv 窗口及 SSM 槽位。
        # 跨设备源存储必须已通过 peer/IPC 映射，批量复制不改变双方页池。
        conv_copies, ssm_copies = [], []
        for src_req, dst_req in zip(src_req_indices, dst_req_indices):
            conv_copies.append((
                remote_dp_bounded_state.conv_state[:, src_req, :, : self.config.conv_width],
                self.conv_state[:, dst_req, :, : self.config.conv_width],
            ))
            ssm_copies.append((
                remote_dp_bounded_state.ssm_state[:, src_req * (remote_dp_bounded_state.mtp_step + 1)],
                self.ssm_state[:, dst_req * (self.mtp_step + 1)],
            ))
        copy_bounded_att_tensors(copies=conv_copies)
        copy_bounded_att_tensors(copies=ssm_copies)
        if self.mtp_state_index is not None:
            for dst_req in dst_req_indices:
                self.mtp_state_index[dst_req].fill_(0)

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 PD 传输字段与切片，交由管理器安排其在联合页中的位置。"""
        return self._get_transfer_regions()

    def write_to_pd_buffer(
        self,
        req_idx: int,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """写出 state_token_len 处的 prefill 基准状态；Conv/SSM 不依赖长度，源 MTP 游标须为 0。"""
        assert layout.resource_id == self.resource_id, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        fields = self._runtime_fields(req_idx)
        for copies in bounded_att_buffer_copies(
            page_fields=[fields],
            page_buffers=pd_page_buffers,
            layout=layout,
            page_indices=[pd_page_index],
            write=True,
        ):
            copy_bounded_att_tensors(copies=copies)

    def read_from_pd_buffer(
        self,
        pd_page_buffers: torch.Tensor,
        req_idx: int,
        pd_page_index: int,
        state_token_len: int,
        layout: StateTransferLayout,
    ) -> None:
        """恢复 state_token_len 处的基准状态并规范化控制信息；Conv/SSM 不依赖长度定位。"""
        assert layout.resource_id == self.resource_id, "transfer layout does not describe this resource"
        assert 0 <= layout.byte_start < layout.byte_end, "invalid State byte interval"
        fields = self._runtime_fields(req_idx)
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
            self.resource_id,
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
                axis_slices=(
                    AxisSlice(
                        axis="channel",
                        src_start=local_channel_start,
                        dst_start=target_channel_start,
                        length=local_channel_count,
                    ),
                    AxisSlice(
                        axis="window",
                        src_start=0,
                        dst_start=0,
                        length=config.conv_width,
                    ),
                ),
                write_enabled=write_enabled,
            )
            regions.append(conv_region)
            logger.debug(
                "%s Conv %s: local_channel_start=%d, global_channel_start=%d, channels=%d, replicas=%d, write_enabled=%s",
                self.resource_id,
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
            axis_slices=(
                AxisSlice(
                    axis="head",
                    src_start=0,
                    dst_start=head_start,
                    length=config.num_v_heads,
                ),
                AxisSlice(
                    axis="key_dim",
                    src_start=0,
                    dst_start=0,
                    length=config.key_dim,
                ),
                AxisSlice(
                    axis="value_dim",
                    src_start=0,
                    dst_start=0,
                    length=config.value_dim,
                ),
            ),
            write_enabled=write_enabled,
        )
        regions.append(ssm_region)
        logger.debug(
            "%s transfer regions built: SSM head_start=%d, heads=%d, write_enabled=%s, SSM bytes=[%d, %d)",
            self.resource_id,
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
        assert pool is not None, f"{page_kind} pages are disabled for {self.resource_id}"
        return pool

    def _runtime_fields(self, req: int):
        """返回请求基准位置的 Conv/SSM 视图，供恢复和传输使用。"""
        # TODO：当前 DP/PD 不会发生在 MTP 提交后，源状态游标为 0。
        # 若以后支持该场景，需要按已接受游标选取 Conv 窗口及 SSM 槽位。
        return self._fields(
            self.conv_state[:, req, :, : self.config.conv_width], self.ssm_state[:, req * (self.mtp_step + 1)]
        )

    @staticmethod
    def _fields(conv: torch.Tensor, ssm: torch.Tensor):
        return {
            "conv": TensorField(tensor=conv, axes=("layer", "channel", "window")),
            "ssm": TensorField(tensor=ssm, axes=("layer", "head", "key_dim", "value_dim")),
        }

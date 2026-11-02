from __future__ import annotations

from lightllm.common.attention_state.utils import weak_cached_method
from typing import TYPE_CHECKING, List
from math import prod

import torch

from lightllm.utils.log_utils import init_logger

from ..utils import deepcopy_result, dtype_name
from ..types import AxisMapping, TensorField, TransferRegion
from .base_state import BoundedAttentionState
from .conv_ssm_types import ConvSSMStateConfig
from .copy_kernel import copy_bounded_att_tensors
from .copy_plan import BatchedStateView

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import PageKind

logger = init_logger(__name__)


class ConvSSMState(BoundedAttentionState):

    algorithm: str
    pool_type: type
    runtime_storage_fields = ("conv_state", "ssm_state", "mtp_state_index")

    @property
    def resource_kind(self):
        return self.algorithm

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
        self.state_name: str | None = None  # 由 AttentionStateManager.register_state() 绑定时设置。
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
            (
                storage_layer_count,
                request_capacity * (mtp_step + 1),
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

    def get_state_type(self) -> str:

        return (
            f"{self.algorithm}_conv_{dtype_name(dtype=self.config.conv_dtype)}"
            f"_ssm_{dtype_name(dtype=self.config.ssm_dtype)}"
        )

    def get_checkpoint_bytes(self):
        c = self.config
        return len(c.logical_layer_ids) * (
            c.conv_channels * c.conv_width * c.conv_dtype.itemsize
            + c.num_v_heads * c.key_dim * c.value_dim * c.ssm_dtype.itemsize
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
            self.pool_type(
                size=big_page_capacity,
                config=self.config,
                device=device,
                pin_memory=pin_memory,
            )
            if big_page_capacity
            else None
        )
        self.small_pages_pool = (
            self.pool_type(
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

    def save_bounded_att_runtime_to_pages(
        self,
        req_indices: List[int],
        state_token_lens: List[int],
        page_kind: PageKind,
        page_slot_ids: List[int],
    ) -> None:
        """保存各请求指定绝对 token 长度处的基准状态；Conv/SSM 不依赖长度定位。"""
        assert isinstance(req_indices, list) and isinstance(
            page_slot_ids, list
        ), "request and slot indices must be lists"
        assert len(req_indices) == len(state_token_lens) == len(page_slot_ids), "request, length and page counts differ"
        pool = self._pool(page_kind)
        offsets = self._accepted_byte_offsets(req_indices, self.conv_state.device)
        conv_copies, ssm_copies = [], []
        for req_idx, slot_id in zip(req_indices, page_slot_ids):
            conv_copies.append((self.conv_state[:, req_idx, :, : self.config.conv_width], pool.conv_state[slot_id]))
            ssm_copies.append((self.ssm_state[:, req_idx * (self.mtp_step + 1)], pool.ssm_state[slot_id]))
        copy_bounded_att_tensors(copies=conv_copies, source_byte_offsets=offsets["conv"])
        copy_bounded_att_tensors(copies=ssm_copies, source_byte_offsets=offsets["ssm"])

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

    @deepcopy_result
    @weak_cached_method
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回已接受 Conv/SSM 快照的字段区间与切片，供 CPU 页布局和容量计算使用。

        region.shape 描述 CPU 页中的全局字段；axis_mappings 将本地页池的 TP 分片映射到该字段。
        起止位置相对本 State，由管理器平移到 CPU 联合页；不包含 MTP 候选区或游标。
        """
        return self._get_transfer_regions()

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @deepcopy_result
    @weak_cached_method
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """返回 PD 传输字段与切片，交由管理器安排其在联合页中的位置。"""
        return self._get_transfer_regions()

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

    def _accepted_byte_offsets(self, requests, device):
        offsets = self.get_runtime_transfer_view().offsets(requests, device)
        return dict(conv=offsets.get("conv"), ssm=offsets.get("ssm"))

    def get_runtime_transfer_view(self):
        return BatchedStateView(
            (
                (
                    "conv",
                    TensorField(
                        self.conv_state.movedim(1, 0)[..., : self.config.conv_width],
                        ("slot", "layer", "channel", "window"),
                    ),
                ),
                (
                    "ssm",
                    TensorField(
                        self.ssm_state[:, :: self.mtp_step + 1].movedim(1, 0),
                        ("slot", "layer", "head", "key_dim", "value_dim"),
                    ),
                ),
            ),
            self.mtp_state_index,
            (
                ("conv", self.conv_state.element_size()),
                ("ssm", self.ssm_state.stride(1) * self.ssm_state.element_size()),
            ),
        )

    def get_snapshot_transfer_view(self, page_kind):
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        if pool is None:
            return BatchedStateView(())
        return BatchedStateView(
            (
                ("conv", TensorField(pool.conv_state, ("slot", "layer", "channel", "window"))),
                ("ssm", TensorField(pool.ssm_state, ("slot", "layer", "head", "key_dim", "value_dim"))),
            )
        )

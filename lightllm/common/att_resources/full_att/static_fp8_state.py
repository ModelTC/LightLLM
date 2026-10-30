from __future__ import annotations

from typing import TYPE_CHECKING, List
from dataclasses import dataclass
from math import prod

import torch

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from ..types import AxisSlice, TransferRegion
from .base_state import FullAttentionConfig, FullAttentionState
from .static_fp8_transfer import transfer_static_fp8_pages

if TYPE_CHECKING:
    from lightllm.common.att_resources.types import StateTransferLayout


logger = init_logger(name=__name__)


@dataclass(frozen=True)
class StaticFP8AttentionConfig(FullAttentionConfig):
    # per_tensor：每层 K/V 各一个固定 scale；per_head：每层、每个本地 K/V head 各一个。
    # 标定参数不随 token 增长，也不作为动态 KV 字段写入 CPU/PD 页。
    scale_mode: str = "per_tensor"
    # 标定数据的身份标记，用于 DP 复制时检查编码是否兼容；实际 scale Tensor 由 State 持有。
    calibration_id: str = ""

    def __post_init__(self) -> None:
        """创建配置时校验通用 TP 约束、标定方式及身份。"""
        super().__post_init__()
        assert self.scale_mode in ("per_tensor", "per_head") and self.calibration_id, (
            "static FP8 requires a scale mode and calibration identity"
        )


class StaticFP8FullAttentionState(FullAttentionState):
    """使用静态标定参数的 FP8 KV 存储，固定 scale 随 State 初始化，不逐 token 搬运。"""


    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        layout_config: StaticFP8AttentionConfig,
        page_size: int = 1,
        *,
        device: str | torch.device = "cuda",
        scales: torch.Tensor,
        q_scales: torch.Tensor | None = None,
    ) -> None:
        """按静态 FP8 配置准备 KV 存储，并保留传入的标定参数。"""
        # 标定 Tensor 由 State 持有，在接收构造参数时校验，不重复放入 config。
        heads = 1 if layout_config.scale_mode == "per_tensor" else layout_config.head_num
        assert scales is not None and scales.shape == (len(layout_config.logical_layer_groups), 2 * heads), (
            "calibration scales do not match KV layers and heads"
        )
        assert scales.dtype == torch.float32 and bool((torch.isfinite(input=scales) & (scales > 0)).all()), (
            "calibration scales must be finite positive float32 values"
        )
        if q_scales is not None:
            assert q_scales.shape == (len(layout_config.logical_layer_groups), layout_config.head_num) and (
                q_scales.dtype == torch.float32
            ), "Q calibration does not match layers and heads"
        self.scales = scales
        self.q_scales = q_scales
        self.cpu_scales = None  # 实际存储初始化时准备计算 backend 使用的 CPU 标定值。
        self.config = layout_config
        self.resource_id: str | None = None  # 由 manager 绑定模型角色时设置。
        # 外层下标是物理层位置；组内多个逻辑层映射到同一份 KV/量化字段存储。
        self._logical_to_storage_layer = {
            logical_layer_id: storage_layer_index
            for storage_layer_index, logical_layer_group in enumerate(layout_config.logical_layer_groups)
            for logical_layer_id in logical_layer_group
        }
        self.page_size = page_size
        self.device = torch.device(device=device)
        self.capacity = 0
        self.k_storage: torch.Tensor | None = None
        self.v_storage: torch.Tensor | None = None
        self.storage_dtype = torch.uint8

    def get_state_type(self) -> str:
        """返回包含 per-tensor 或 per-head 标定方式的 FP8 方案标记。"""
        return "gqa_fp8_static_" + self.config.scale_mode

    def initialize_storage(self, token_capacity: int) -> None:
        """绑定 KV 与真实标定参数，并预备 FlashInfer 使用的 CPU scale。"""
        self.capacity = token_capacity
        shape = self._storage_shape()
        self.k_storage = torch.empty(size=shape, dtype=self.storage_dtype, device=self.device)
        self.v_storage = torch.empty(size=shape, dtype=self.storage_dtype, device=self.device)
        # meta 准备阶段保留真实标定值，实际分配时再迁移到 KV 所在设备。
        if self.k_storage.device.type != "meta":
            self.scales = self.scales.to(device=self.k_storage.device)
            if self.q_scales is not None:
                self.q_scales = self.q_scales.to(device=self.k_storage.device)
        # 提前准备 CPU scale，避免计算热路径同步设备。
        self.cpu_scales = self.scales.detach().cpu().numpy()

    def get_token_bytes(self) -> int:
        """由配置累计当前 TP rank 的 K/V 单 token 开销，无需分配存储。"""
        config = self.config
        return (
            len(config.logical_layer_groups) * 2 * config.head_num * config.head_dim
            * self.storage_dtype.itemsize
        )

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    def has_logical_layer(self, logical_layer_id: int) -> bool:
        """查询逻辑层归属；共享同一物理层的逻辑层均返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_layer_view(self, logical_layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑层号返回 K、V 视图，head 数使用当前 TP 分片配置。"""
        storage_layer_index = self._logical_to_storage_layer[logical_layer_id]
        return self.k_storage[storage_layer_index], self.v_storage[storage_layer_index]

    def write_full_att_kv(self, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """按固定 per-tensor 或 per-head scale 量化输入 K/V，写入 FP8 编码。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = self._logical_to_storage_layer[logical_layer_id]
        if not len(token_indices):
            return
        assert self.k_storage.is_cuda, "KV writes require CUDA storage"
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8 import destindex_copy_kv_fp8

        destindex_copy_kv_fp8(
            KV=kv,
            DestLoc=token_indices,
            scales=self.scales[storage_layer_index],
            Out=self.k_storage[storage_layer_index].view(torch.float8_e4m3fn),
            is_per_tensor_quant=self.config.scale_mode == "per_tensor",
            Out_v=self.v_storage[storage_layer_index].view(torch.float8_e4m3fn),
        )

    def copy_full_att_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """复制完整 token 数据；索引使用 Tensor，源与目标位置由调用方保证不重叠。"""
        assert src_indices.is_cuda and dst_indices.is_cuda, "token indices must reside on CUDA"
        assert src_indices.ndim == dst_indices.ndim == 1 and src_indices.numel() == dst_indices.numel(), (
            "token copies require equal one-dimensional index tensors"
        )
        assert src_indices.dtype in (torch.int32, torch.int64) and dst_indices.dtype in (torch.int32, torch.int64), (
            "token indices must be integer tensors"
        )
        assert self.k_storage.is_cuda, "token copies require CUDA storage"
        if not src_indices.numel():
            return
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        # 直接使用调用方提供的 CUDA 索引，保留原 dtype 和 stride。
        # 按字节直接从源槽位写到目标槽位，覆盖量化 scale 和 packed padding，不创建中间 KV。
        # 调用方保证源、目标集合不相交，且目标索引不重复，避免并行读写竞争。
        # TODO：将 FP8 K/V 原始编码 的同索引 token 复制融合为一次 kernel 发射，
        # 保持原始字节编码及各字段 stride，仍要求源目标不重叠、目标索引不重复。
        copy_kv_buffer_to_kv_buffer(
            src_mem_index=src_indices,
            dst_mem_index=dst_indices,
            kv_buffer=self.k_storage,
        )
        copy_kv_buffer_to_kv_buffer(
            src_mem_index=src_indices,
            dst_mem_index=dst_indices,
            kv_buffer=self.v_storage,
        )

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 CPU 页大小返回字段区间与 TP 切片，作为 CPU 页容量计算依据。

        覆盖 KV、动态量化参数及 indexer 有效字段，不包含 GPU 行 padding 或静态标定参数。
        """
        page_tokens = get_env_start_args().cpu_cache_token_page_size
        return self._get_transfer_regions(page_tokens=page_tokens)

    def write_to_cpu(
        self,
        token_indices: torch.Tensor,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        *,
        layout: StateTransferLayout,
        cpu_page_readies: list[bool] | None = None,
    ) -> None:
        """按布局直接写出有效 token 及配套字段，不创建中间 KV 存储。"""
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        transfer_static_fp8_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
            token_indices=token_indices,
            page_buffers=cpu_page_buffers,
            page_indices=cpu_page_indices,
            layout=layout,
            write=True,
            page_readies=cpu_page_readies,
        )

    def read_from_cpu(
        self,
        cpu_page_buffers: torch.Tensor,
        cpu_page_indices: List[int],
        token_indices: torch.Tensor,
        *,
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接恢复有效 token 及配套字段，保留未参与的存储区域。"""
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        transfer_static_fp8_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
            token_indices=token_indices,
            page_buffers=cpu_page_buffers,
            page_indices=cpu_page_indices,
            layout=layout,
            write=False,
        )

    # ========================================================================
    # 四、DP 迁移
    # ========================================================================

    def copy_tokens_from_other_dp(
        self,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        remote_dp_full_att_state: StaticFP8FullAttentionState,
    ) -> None:
        """先校验标定方式与标定身份，再复制兼容的 FP8 KV 编码。"""
        assert (
            remote_dp_full_att_state.config.scale_mode,
            remote_dp_full_att_state.config.calibration_id,
        ) == (
            self.config.scale_mode,
            self.config.calibration_id,
        ), "incompatible static FP8 calibration"
        assert type(remote_dp_full_att_state) is type(self) and remote_dp_full_att_state.config == self.config, (
            "DP copy requires the same encoding and equivalent TP state configurations"
        )
        assert src_token_indices.is_cuda and dst_token_indices.is_cuda, "DP token indices must reside on CUDA"
        assert src_token_indices.numel() == dst_token_indices.numel(), "source and destination token counts differ"
        # TODO：改用 Triton kernel 按源/目标 token 索引直接复制，融合字段搬运，去掉中间 gather 存储；
        # 跨设备时源存储需可通过 peer/IPC 访问，并遵循调用方的 stream/event 依赖。
        # 同一 TP 分片复制完整 token；调用方提供各自设备上的 CUDA 索引，目标索引须为 int64。
        src_k = remote_dp_full_att_state.k_storage.index_select(dim=1, index=src_token_indices)
        dst_k = src_k.to(device=self.k_storage.device, non_blocking=True)
        self.k_storage.index_copy_(dim=1, index=dst_token_indices, source=dst_k)
        src_v = remote_dp_full_att_state.v_storage.index_select(dim=1, index=src_token_indices)
        dst_v = src_v.to(device=self.v_storage.device, non_blocking=True)
        self.v_storage.index_copy_(dim=1, index=dst_token_indices, source=dst_v)

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 PD 页大小返回字段与切片，交由管理器安排其在联合页中的位置。"""
        page_tokens = get_env_start_args().pd_kv_page_size
        return self._get_transfer_regions(page_tokens=page_tokens)

    def write_to_pd_buffer(
        self,
        token_indices: torch.Tensor,
        pd_page_buffers: torch.Tensor,
        *,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接写出有效 token 及配套字段，不创建中间 KV 存储。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_static_fp8_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
            token_indices=token_indices,
            page_buffers=pd_page_buffers,
            page_indices=[pd_page_index],
            layout=layout,
            write=True,
        )

    def read_from_pd_buffer(
        self,
        pd_page_buffers: torch.Tensor,
        token_indices: torch.Tensor,
        *,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接恢复有效 token 及配套字段，保留未参与的存储区域。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_static_fp8_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
            token_indices=token_indices,
            page_buffers=pd_page_buffers,
            page_indices=[pd_page_index],
            layout=layout,
            write=False,
        )

    # ========================================================================
    # 六、私有辅助实现
    # ========================================================================

    def _get_transfer_regions(self, page_tokens: int) -> tuple[TransferRegion, ...]:
        """依次排列 KV、动态量化等字段，描述全局页形状和本地 TP 分片。"""
        config = self.config
        byte_offset = 0
        regions = []
        logger.debug(
            "%s building transfer regions: TP=%d/%d, tokens_per_page=%d",
            self.resource_id, config.tp_rank, config.tp_world_size, page_tokens,
        )
        # 阶段一：按本方案固定字段计算形状和 dtype，不分配 meta 存储。
        field_specs = (
            ("k", self.storage_dtype, config.head_dim, "dim"),
            ("v", self.storage_dtype, config.head_dim, "dim"),
        )
        for field_name, field_dtype, field_width, last_axis in field_specs:
            axes = ("layer", "token", "head", last_axis)
            local_shape = (len(config.logical_layer_groups), page_tokens, config.head_num, field_width)
            global_shape = list(local_shape)
            global_shape[axes.index("token")] = page_tokens
            head_axis = axes.index("head")
            # K/V 各自是独立字段：这里读取的是单个字段的本地 head 数。
            local_head_count = global_shape[head_axis]
            # 外部页保存该字段未做 TP 切分的完整 heads，TP 副本只占一份区域。
            # K/V 各生成一个 region 并分别累加字节数，因此这里不能再乘 2；动态 scale 同理。
            # 例如全局 H=8、TP=2：本地 k/v 各 4 heads，外部 k/v 区域各 8 heads。
            global_shape[head_axis] = config.global_head_num

            # 阶段二：确定本地 heads 的全局起点；同组副本只有首 rank 写出，所有副本都可读回。
            # replica_count 表示同一份 head 分片被多少个 ranks 复制：纯切分时为 1，全复制时为 TP。
            replica_count = config.tp_world_size * local_head_count // config.global_head_num
            # 将组内 rank 换算为去重后的分片编号，再定位当前字段在全局 head 轴上的起点。
            head_start = config.tp_rank // replica_count * local_head_count
            # 副本共享同一外部区域，只让组内第一个 rank 写出，避免重复写入；读取不受此标记限制。
            write_enabled = config.tp_rank % replica_count == 0
            axis_slices = tuple(
                AxisSlice(
                    axis=axis,
                    src_start=0,
                    dst_start=head_start if axis == "head" else 0,
                    length=dimension,
                )
                for axis, dimension in zip(axes, local_shape)
                if axis not in ("layer", "token")
            )

            # 阶段三：按字段 dtype 对齐起点，记录本字段区间，再推进下一个字段的字节偏移。
            element_bytes = field_dtype.itemsize
            byte_offset = (byte_offset + element_bytes - 1) // element_bytes * element_bytes
            field_region = TransferRegion(
                field=field_name,
                byte_start=byte_offset,
                byte_end=byte_offset + prod(global_shape) * element_bytes,
                dtype=field_dtype,
                axes=axes,
                shape=tuple(global_shape),
                axis_slices=axis_slices,
                write_enabled=write_enabled,
            )
            regions.append(field_region)
            byte_offset = field_region.byte_end
            logger.debug(
                "%s field %s: head_start=%d, heads=%d, replicas=%d, write_enabled=%s, bytes=[%d, %d)",
                self.resource_id, field_name, head_start, local_head_count, replica_count,
                write_enabled, field_region.byte_start, field_region.byte_end,
            )
        logger.debug(
            "%s transfer regions built: fields=%d, total_bytes=%d", self.resource_id, len(regions), byte_offset
        )
        return tuple(regions)

    def _storage_shape(self) -> tuple[int, ...]:
        return (
            len(self.config.logical_layer_groups),
            self.capacity + self.page_size,
            self.config.head_num,
            self.config.head_dim,
        )

from __future__ import annotations

from typing import TYPE_CHECKING, List
from dataclasses import dataclass
from math import prod

import torch

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from ..utils import dtype_name
from ..types import AxisSlice, TransferRegion
from .base_state import FullAttentionConfig, FullAttentionState
from .group_quantized_transfer import transfer_group_quantized_pages
from .dp_transfer import copy_group_quantized_tokens

if TYPE_CHECKING:
    from lightllm.common.att_resources.types import StateTransferLayout


logger = init_logger(name=__name__)


@dataclass(frozen=True)
class GroupQuantizedAttentionConfig(FullAttentionConfig):
    # 每个 KV 元素的量化位数，仅支持 4/8；INT4 将两个编码值打包到一个字节。
    bits: int = 8
    # 沿每个 head 的 dim 轴分组，每个 token、每组共用一个动态 scale；须整除 head_dim。
    # K/V 分别保存 scale，精度由继承的 dtype 指定。
    group_size: int = 8

    def __post_init__(self) -> None:
        """创建配置时校验通用 TP 约束及量化分组要求。"""
        super().__post_init__()
        assert self.bits in (4, 8) and self.group_size >= 2 and not (
            self.group_size & (self.group_size - 1)
        ), "INT4/INT8 quantization requires a power-of-two group size"
        assert self.head_dim % self.group_size == 0 and self.dtype in (
            torch.float16, torch.bfloat16, torch.float32
        ), "group scales require a floating dtype and complete groups"


class GroupQuantizedFullAttentionState(FullAttentionState):
    """管理 INT4/INT8 KV 与逐 token 分组 scale，复制和传输始终覆盖二者。"""


    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        layout_config: GroupQuantizedAttentionConfig,
        page_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        """按分组量化配置准备 KV 与动态 scale 存储。"""
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
        self.storage_dtype = torch.int8
        self.k_scales: torch.Tensor | None = None
        self.v_scales: torch.Tensor | None = None

    def get_state_type(self) -> str:
        """返回量化位数、分组大小及 scale 精度对应的存储方案标记。"""
        return f"gqa_int{self.config.bits}_group{self.config.group_size}_{dtype_name(dtype=self.config.dtype)}_scale"

    def initialize_storage(self, token_capacity: int) -> None:
        """按统一 token 容量分配独立的量化 KV 与配套动态 scale。"""
        self.capacity = token_capacity
        shape = self._storage_shape()
        self.k_storage = torch.empty(size=shape, dtype=self.storage_dtype, device=self.device)
        self.v_storage = torch.empty(size=shape, dtype=self.storage_dtype, device=self.device)
        shape = (
            len(self.config.logical_layer_groups), token_capacity + self.page_size,
            self.config.head_num, self.config.head_dim // self.config.group_size,
        )
        self.k_scales = torch.empty(size=shape, dtype=self.config.dtype, device=self.device)
        self.v_scales = torch.empty(size=shape, dtype=self.config.dtype, device=self.device)

    def get_token_bytes(self) -> int:
        """累计量化 KV 编码与逐 token 分组 scale 的开销，无需分配存储。"""
        config = self.config
        head_bytes = config.head_dim * config.bits // 8
        head_bytes += config.head_dim // config.group_size * config.dtype.itemsize
        return len(config.logical_layer_groups) * 2 * config.head_num * head_bytes

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，共享层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_storage_layer_index(self, logical_layer_id: int) -> int:
        """按逻辑层号查询存储及配套字段的物理行；共享逻辑层返回同一下标。"""
        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """返回该层 K/V 编码数据及各自的分组 scale 视图。"""
        storage_layer_index = self.get_storage_layer_index(logical_layer_id=logical_layer_id)
        return (self.k_storage[storage_layer_index], self.k_scales[storage_layer_index]), (self.v_storage[storage_layer_index], self.v_scales[storage_layer_index])

    def write_full_att_kv(self, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """量化输入浮点 K/V，并将编码数据和分组 scale 写入相同 token 位置。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = self.get_storage_layer_index(logical_layer_id=logical_layer_id)
        if not len(token_indices):
            return
        assert self.k_storage.is_cuda, "KV writes require CUDA storage"
        # TODO：融合 K/V 两次量化写入，同时输出各自编码和动态 scale，保持分组及舍入语义。
        # K/V 分别写入独立存储，各自携带同一 token 的动态 scale。
        for offset, storage, scales_buffer in (
            (0, self.k_storage, self.k_scales),
            (self.config.head_num, self.v_storage, self.v_scales),
        ):
            values = kv[:, offset:offset + self.config.head_num]
            if self.config.bits == 4:
                from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import destindex_copy_int4kv

                destindex_copy_int4kv(
                    KV=values,
                    DestLoc=token_indices,
                    KV_buffer=storage[storage_layer_index],
                    KV_scale_buffer=scales_buffer[storage_layer_index],
                    quant_group_size=self.config.group_size,
                )
            else:
                from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv import destindex_copy_quantize_kv

                destindex_copy_quantize_kv(
                    K=values,
                    DestLoc=token_indices,
                    Out=storage[storage_layer_index],
                    Out_scale=scales_buffer[storage_layer_index],
                    quant_group_dim=self.config.group_size,
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
        # TODO：将 K/V 编码及各自的动态 scale 的同索引 token 复制融合为一次 kernel 发射，
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
        copy_kv_buffer_to_kv_buffer(
            src_mem_index=src_indices,
            dst_mem_index=dst_indices,
            kv_buffer=self.k_scales,
        )
        copy_kv_buffer_to_kv_buffer(
            src_mem_index=src_indices,
            dst_mem_index=dst_indices,
            kv_buffer=self.v_scales,
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
        layout: StateTransferLayout,
        cpu_page_readies: list[bool] | None = None,
    ) -> None:
        """按布局直接写出有效 token 及配套字段，不创建中间 KV 存储。"""
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        transfer_group_quantized_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
            k_scales=self.k_scales,
            v_scales=self.v_scales,
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
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接恢复有效 token 及配套字段，保留未参与的存储区域。"""
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        transfer_group_quantized_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
            k_scales=self.k_scales,
            v_scales=self.v_scales,
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
        remote_dp_full_att_state: GroupQuantizedFullAttentionState,
    ) -> None:
        """校验量化位数和分组大小，再复制 KV 与动态 scale。"""
        assert (
            remote_dp_full_att_state.config.bits,
            remote_dp_full_att_state.config.group_size,
        ) == (
            self.config.bits, self.config.group_size
        ), "incompatible group quantization"
        assert type(remote_dp_full_att_state) is type(self) and remote_dp_full_att_state.config == self.config, (
            "DP copy requires the same encoding and equivalent TP state configurations"
        )
        assert src_token_indices.is_cuda and dst_token_indices.is_cuda, "DP token indices must reside on CUDA"
        assert src_token_indices.numel() == dst_token_indices.numel(), "source and destination token counts differ"
        copy_group_quantized_tokens(
            src_k_storage=remote_dp_full_att_state.k_storage,
            src_v_storage=remote_dp_full_att_state.v_storage,
            src_k_scales=remote_dp_full_att_state.k_scales,
            src_v_scales=remote_dp_full_att_state.v_scales,
            dst_k_storage=self.k_storage,
            dst_v_storage=self.v_storage,
            dst_k_scales=self.k_scales,
            dst_v_scales=self.v_scales,
            src_token_indices=src_token_indices,
            dst_token_indices=dst_token_indices,
        )

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
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接写出有效 token 及配套字段，不创建中间 KV 存储。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_group_quantized_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
            k_scales=self.k_scales,
            v_scales=self.v_scales,
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
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接恢复有效 token 及配套字段，保留未参与的存储区域。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_group_quantized_pages(
            k_storage=self.k_storage,
            v_storage=self.v_storage,
            k_scales=self.k_scales,
            v_scales=self.v_scales,
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
            self.resource_id,
            config.tp_rank,
            config.tp_world_size,
            page_tokens,
        )
        # 阶段一：按本方案固定字段计算形状和 dtype，不分配 meta 存储。
        field_specs = (
            ("k", torch.int8, config.head_dim * config.bits // 8, "dim"),
            ("v", torch.int8, config.head_dim * config.bits // 8, "dim"),
            ("k_scale", config.dtype, config.head_dim // config.group_size, "group"),
            ("v_scale", config.dtype, config.head_dim // config.group_size, "group"),
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
                self.resource_id,
                field_name,
                head_start,
                local_head_count,
                replica_count,
                write_enabled,
                field_region.byte_start,
                field_region.byte_end,
            )
        logger.debug(
            "%s transfer regions built: fields=%d, total_bytes=%d",
            self.resource_id,
            len(regions),
            byte_offset,
        )
        return tuple(regions)

    def _storage_shape(self) -> tuple[int, ...]:
        return (
            len(self.config.logical_layer_groups), self.capacity + self.page_size,
            self.config.head_num, self.config.head_dim * self.config.bits // 8,
        )

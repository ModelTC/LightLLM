from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List
from dataclasses import dataclass
from math import prod

import torch

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from ..utils import deepcopy_result, dtype_name
from ..types import AxisMapping, TransferRegion
from .base_state import FullAttentionConfig, FullAttentionState
from .packed_mla_transfer import transfer_packed_mla_pages
from .dp_transfer import copy_packed_tokens

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import StateTransferLayout


logger = init_logger(name=__name__)


@dataclass(frozen=True)
class PackedMLAAttentionConfig(FullAttentionConfig):
    # MLA KV latent 的有效维度，包含需要保存的 RoPE 部分，不含 indexer、scale 或行 padding。
    kv_dim: int = 576
    # 每个 token 的 indexer key 维度，与 MLA latent 一起存放在同一物理行。
    indexer_dim: int = 128
    # 0 表示不使用 FP8 latent 编码；非零表示需要 FP8 量化的非 RoPE 部分维度。
    # 当前 FP8 实现要求 512 维非 RoPE latent，剩余 64 维 RoPE 以 BF16 保存。
    fp8_nope_dim: int = 0
    # FP8 非 RoPE latent 的分组量化宽度，当前 kernel 要求每 128 个元素对应一个 scale。
    quant_group_size: int = 128

    def __post_init__(self) -> None:
        """创建配置时校验通用 TP 约束、MLA 编码及 packed 行宽。"""
        super().__post_init__()
        assert self.head_num == 1 and self.indexer_dim > 0, (
            "packed MLA requires one replicated latent and a positive indexer dimension"
        )
        if self.fp8_nope_dim:
            assert (self.fp8_nope_dim, self.kv_dim, self.quant_group_size, self.dtype) == (
                512,
                576,
                128,
                torch.uint8,
            ), "FP8 MLA kernel requires 512 latent dimensions, 64 RoPE dimensions and groups of 128"
            payload_bytes = 656 + self.indexer_dim + 4
        else:
            assert self.dtype in (torch.float16, torch.bfloat16), "unquantized packed MLA requires FP16 or BF16"
            payload_bytes = self.kv_dim * self.dtype.itemsize + self.indexer_dim + 4
        assert self.head_dim * self.dtype.itemsize == (payload_bytes + 15) // 16 * 16, (
            "packed token width must include aligned KV and indexer storage"
        )


@dataclass(frozen=True)
class PackedMLALayerView:
    kv: torch.Tensor
    indexer_k: torch.Tensor


class PackedMLAFullAttentionState(FullAttentionState):
    """在同一物理 token 行中存储 MLA latent、量化参数及稀疏 indexer key/scale。"""


    # ========================================================================
    # 一、资源定义与初始化
    # ========================================================================

    def __init__(
        self,
        layout_config: PackedMLAAttentionConfig,
        page_size: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        """按 packed MLA 配置准备 KV 与 indexer 存储。"""
        self.config = layout_config
        self.storage_dtype = layout_config.dtype  # 实际 KV 编码类型；量化子类显式设置，区别于 scale 精度。
        self.state_name: str | None = None  # 由 manager 绑定模型角色时设置。
        # 外层下标是物理层位置；组内多个逻辑层映射到同一份 KV/量化字段存储。
        self._logical_to_storage_layer = {
            logical_layer_id: storage_layer_index
            for storage_layer_index, logical_layer_group in enumerate(layout_config.logical_layer_groups)
            for logical_layer_id in logical_layer_group
        }
        self.page_size = page_size
        self.device = torch.device(device=device)
        self.capacity = 0
        self.storage: torch.Tensor | None = None

    def get_state_type(self) -> str:
        """返回 packed MLA 编码方案标记，区分 FP8 与原始精度。"""
        encoding = f"fp8_group{self.config.quant_group_size}" if self.config.fp8_nope_dim else dtype_name(dtype=self.config.dtype)
        return "mla_packed_" + encoding

    def allocate_storage(self, token_capacity: int) -> None:
        """按统一 token 容量分配本 State 的独立 KV 存储与 HOLD 页。"""
        self.capacity = token_capacity
        shape = self._storage_shape()
        self.storage = torch.empty(size=shape, dtype=self.storage_dtype, device=self.device)

    def get_token_bytes(self) -> int:
        """由配置计算当前 TP rank 的单 token 开销，包含 packed 行 padding，无需分配存储。"""
        config = self.config
        return (
            len(config.logical_layer_groups) * config.head_num * config.head_dim
            * self.storage_dtype.itemsize
        )

    @property
    def kv_nbytes(self) -> int:
        """返回一行中 KV 区域的字节数，包含该编码需要的 scale 与 RoPE 数据。"""
        return 656 if self.config.fp8_nope_dim else self.config.kv_dim * self.config.dtype.itemsize

    @property
    def indexer_offset(self) -> int:
        """返回一行中 indexer key 区域的字节偏移。"""
        if self.config.fp8_nope_dim:
            return self.kv_nbytes
        return self.config.head_dim * self.config.dtype.itemsize - self.config.indexer_dim - 4

    # ========================================================================
    # 二、Full attention：计算视图、KV 写入与完整 token 复制
    # ========================================================================

    def contains_logical_layer(self, logical_layer_id: int) -> bool:
        """判断指定逻辑层是否属于当前 State 的层映射，共享层同样返回 True。"""
        return logical_layer_id in self._logical_to_storage_layer

    def get_storage_layer_index(self, logical_layer_id: int) -> int:
        """按逻辑层号查询存储及配套字段的物理行；共享逻辑层返回同一下标。"""
        return self._logical_to_storage_layer[logical_layer_id]

    def get_layer_view(self, logical_layer_id: int) -> PackedMLALayerView:
        """分别返回 KV 与 packed indexer key/scale 的视图，保持原行步长。"""
        row = self.storage[self.get_storage_layer_index(logical_layer_id=logical_layer_id)]
        raw = row.view(torch.uint8)
        kv = raw[..., : self.kv_nbytes] if self.config.fp8_nope_dim else row[..., : self.config.kv_dim]
        indexer = raw[..., self.indexer_offset : self.indexer_offset + self.config.indexer_dim + 4]
        return PackedMLALayerView(kv=kv, indexer_k=indexer)

    def write_full_att_kv(self, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """写入 MLA KV 及其量化参数，保留同一 token 行中的 indexer 区域。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = self.get_storage_layer_index(logical_layer_id=logical_layer_id)
        if not len(token_indices):
            return
        assert self.storage.is_cuda, "KV writes require CUDA storage"
        if self.config.fp8_nope_dim:
            from lightllm.common.basemodel.triton_kernel.kv_copy.destindex_copy_kv_flashmla_fp8 import (
                destindex_copy_kv_flashmla_fp8,
            )

            raw = self.storage[storage_layer_index]
            destindex_copy_kv_flashmla_fp8(
                KV_nope=kv[..., :512],
                KV_rope=kv[..., 512:],
                DestLoc=token_indices,
                O_nope=raw[..., :512].view(torch.float8_e4m3fn),
                O_scale=raw[..., 512:528].view(torch.float32),
                O_rope=raw[..., 528:656].view(torch.bfloat16),
            )
        else:
            from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

            destindex_copy_kv(K=kv, DestLoc=token_indices, Out=self.storage[storage_layer_index, ..., : self.config.kv_dim])

    def copy_full_att_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """复制完整 token 数据；索引使用 Tensor，源与目标位置由调用方保证不重叠。"""
        assert src_indices.is_cuda and dst_indices.is_cuda, "token indices must reside on CUDA"
        assert src_indices.ndim == dst_indices.ndim == 1 and src_indices.numel() == dst_indices.numel(), (
            "token copies require equal one-dimensional index tensors"
        )
        assert src_indices.dtype in (torch.int32, torch.int64) and dst_indices.dtype in (torch.int32, torch.int64), (
            "token indices must be integer tensors"
        )
        assert self.storage.is_cuda, "token copies require CUDA storage"
        if not src_indices.numel():
            return
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        # 直接使用调用方提供的 CUDA 索引，保留原 dtype 和 stride。
        # 按字节直接从源槽位写到目标槽位，覆盖量化 scale 和 packed padding，不创建中间 KV。
        # 调用方保证源、目标集合不相交，且目标索引不重复，避免并行读写竞争。
        copy_kv_buffer_to_kv_buffer(
            src_mem_index=src_indices,
            dst_mem_index=dst_indices,
            kv_buffer=self.storage,
        )

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @lru_cache(maxsize=None)
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
        cpu_page_ready_flags: list[bool] | None = None,
    ) -> None:
        """按布局直接写出有效 token 及配套字段，不创建中间 KV 存储。"""
        assert cpu_page_buffers.device.type == "cpu", "CPU cache buffer must reside on CPU"
        transfer_packed_mla_pages(
            kv_storage=self.storage,
            token_indices=token_indices,
            page_buffers=cpu_page_buffers,
            page_indices=cpu_page_indices,
            layout=layout,
            kv_dim=self.config.kv_dim,
            indexer_offset=self.indexer_offset,
            indexer_dim=self.config.indexer_dim,
            write=True,
            page_readies=cpu_page_ready_flags,
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
        transfer_packed_mla_pages(
            kv_storage=self.storage,
            token_indices=token_indices,
            page_buffers=cpu_page_buffers,
            page_indices=cpu_page_indices,
            layout=layout,
            kv_dim=self.config.kv_dim,
            indexer_offset=self.indexer_offset,
            indexer_dim=self.config.indexer_dim,
            write=False,
        )

    # ========================================================================
    # 四、DP 迁移
    # ========================================================================

    def copy_tokens_from_dp(
        self,
        src_token_indices: torch.Tensor,
        dst_token_indices: torch.Tensor,
        remote_dp_full_att_state: PackedMLAFullAttentionState,
    ) -> None:
        """校验 MLA 编码与区域尺寸，再复制完整 KV 和 indexer 数据。"""
        assert (
            remote_dp_full_att_state.config.fp8_nope_dim,
            remote_dp_full_att_state.config.kv_dim,
            remote_dp_full_att_state.config.indexer_dim,
        ) == (
            self.config.fp8_nope_dim,
            self.config.kv_dim,
            self.config.indexer_dim,
        ), "incompatible packed MLA format"
        assert type(remote_dp_full_att_state) is type(self) and remote_dp_full_att_state.config == self.config, (
            "DP copy requires the same encoding and equivalent TP state configurations"
        )
        assert src_token_indices.is_cuda and dst_token_indices.is_cuda, "DP token indices must reside on CUDA"
        assert src_token_indices.numel() == dst_token_indices.numel(), "source and destination token counts differ"
        copy_packed_tokens(
            src_storage=remote_dp_full_att_state.storage,
            dst_storage=self.storage,
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

    def write_to_pd_page(
        self,
        token_indices: torch.Tensor,
        pd_page_buffers: torch.Tensor,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接写出有效 token 及配套字段，不创建中间 KV 存储。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_packed_mla_pages(
            kv_storage=self.storage,
            token_indices=token_indices,
            page_buffers=pd_page_buffers,
            page_indices=[pd_page_index],
            layout=layout,
            kv_dim=self.config.kv_dim,
            indexer_offset=self.indexer_offset,
            indexer_dim=self.config.indexer_dim,
            write=True,
        )

    def read_from_pd_page(
        self,
        pd_page_buffers: torch.Tensor,
        token_indices: torch.Tensor,
        pd_page_index: int,
        layout: StateTransferLayout,
    ) -> None:
        """按布局直接恢复有效 token 及配套字段，保留未参与的存储区域。"""
        assert pd_page_buffers.is_cuda, "PD pages must reside on CUDA"
        transfer_packed_mla_pages(
            kv_storage=self.storage,
            token_indices=token_indices,
            page_buffers=pd_page_buffers,
            page_indices=[pd_page_index],
            layout=layout,
            kv_dim=self.config.kv_dim,
            indexer_offset=self.indexer_offset,
            indexer_dim=self.config.indexer_dim,
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
            self.state_name,
            config.tp_rank,
            config.tp_world_size,
            page_tokens,
        )
        # 阶段一：按本方案固定字段计算形状和 dtype，不分配 meta 存储。
        if config.fp8_nope_dim:
            field_specs = [
                ("kv_nope", torch.uint8, 512, "dim"),
                ("kv_scale", torch.float32, 4, "group"),
                ("kv_rope", torch.bfloat16, 64, "dim"),
            ]
        else:
            field_specs = [("kv", config.dtype, config.kv_dim, "dim")]
        field_specs.extend((
            ("indexer_k", torch.uint8, config.indexer_dim, "dim"),
            ("indexer_scale", torch.float32, 1, "scale"),
        ))
        for field_name, field_dtype, field_width, last_axis in field_specs:
            axes = ("layer", "token", "head", last_axis)
            local_shape = (len(config.logical_layer_groups), page_tokens, config.head_num, field_width)
            global_shape = list(local_shape)
            global_shape[axes.index("token")] = page_tokens
            head_axis = axes.index("head")
            # field 已按 K/V 拆开：这里读取的是单个字段的本地 head 数，不是合并 buffer 的 2H。
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
            axis_mappings = tuple(
                AxisMapping(
                    axis=axis,
                    local_start=0,
                    external_start=head_start if axis == "head" else 0,
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
                axis_mappings=axis_mappings,
                write_enabled=write_enabled,
            )
            regions.append(field_region)
            byte_offset = field_region.byte_end
            logger.debug(
                "%s field %s: head_start=%d, heads=%d, replicas=%d, write_enabled=%s, bytes=[%d, %d)",
                self.state_name,
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
            self.state_name,
            len(regions),
            byte_offset,
        )
        return tuple(regions)

    def _storage_shape(self) -> tuple[int, ...]:
        return (
            len(self.config.logical_layer_groups),
            self.capacity + self.page_size,
            self.config.head_num,
            self.config.head_dim,
        )

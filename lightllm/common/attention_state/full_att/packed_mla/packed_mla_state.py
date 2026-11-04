from __future__ import annotations

from lightllm.common.attention_state.query_cache import cached_query

from typing import TYPE_CHECKING, List, Literal
from dataclasses import dataclass
from math import prod

import torch

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from ...utils import deepcopy_result, dtype_name
from ...types import AxisMapping, TransferRegion
from ..base_state import FullAttentionState
from .packed_mla_transfer import transfer_packed_mla_pages
from ..dp_transfer import copy_packed_tokens

if TYPE_CHECKING:
    from lightllm.common.attention_state.types import StateTransferLayout


logger = init_logger(name=__name__)


@dataclass(frozen=True)
class PackedMLALogicalConfig:
    """带 indexer 的 packed MLA 逻辑配置；不描述实际字节布局。"""

    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 编码前的完整 latent 宽度；具体分块、scale 和 padding 由工厂转换。
    latent_dim: int
    # packed MLA 的逻辑格式；所有选项都包含 indexer。
    format: Literal["packed_fp16", "packed_bf16", "packed_group128_fp8", "packed_group64_e8m0"]
    # indexer key 的逻辑宽度。
    indexer_dim: int = 128
    # latent 中保留原始 RoPE 精度的宽度；NoPE 宽度由 latent_dim - rope_dim 得到。
    rope_dim: int = 64
    tp_rank: int = 0
    tp_world_size: int = 1

    def __post_init__(self) -> None:
        assert self.logical_layer_groups and all(self.logical_layer_groups)
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert len(layers) == len(set(layers)), "logical layers cannot repeat across storage groups"
        assert (
            self.latent_dim > 0
            and 0 <= self.rope_dim <= self.latent_dim
            and self.indexer_dim > 0
            and 0 <= self.tp_rank < self.tp_world_size
        )
        assert self.format in (
            "packed_fp16",
            "packed_bf16",
            "packed_group128_fp8",
            "packed_group64_e8m0",
        ), "unknown packed MLA logical format"


@dataclass(frozen=True)
class PackedMLAAttentionConfig:
    """由 packed 逻辑配置解析出的物理布局，不重复保存逻辑字段。"""

    logical_config: PackedMLALogicalConfig
    quant_group_size: int = 128

    @property
    def logical_layer_groups(self):
        return self.logical_config.logical_layer_groups

    @property
    def latent_dim(self):
        return self.logical_config.latent_dim

    @property
    def kv_encoding(self):
        return {
            "packed_fp16": "fp16",
            "packed_bf16": "bf16",
            "packed_group128_fp8": "fp8_group128",
        }[self.logical_config.format]

    @property
    def indexer_dim(self):
        return self.logical_config.indexer_dim

    @property
    def tp_rank(self):
        return self.logical_config.tp_rank

    @property
    def tp_world_size(self):
        return self.logical_config.tp_world_size

    @property
    def rope_dim(self):
        return self.logical_config.rope_dim

    @property
    def dtype(self) -> torch.dtype:
        """返回编码对应的逻辑存储 dtype；FP8 的物理承载仍由 State 内部使用 uint8。"""
        if self.kv_encoding == "fp16":
            return torch.float16
        if self.kv_encoding == "bf16":
            return torch.bfloat16
        return torch.float8_e4m3fn

    @property
    def kv_bytes(self) -> int:
        """按逻辑 NoPE/RoPE 宽度和 scale 数量计算单行 KV 字节数。"""
        if self.kv_encoding == "fp8_group128":
            nope_dim = self.latent_dim - self.rope_dim
            scale_count = nope_dim // self.quant_group_size
            return nope_dim + scale_count * torch.float32.itemsize + self.rope_dim * torch.bfloat16.itemsize
        return self.latent_dim * self.dtype.itemsize

    @property
    def token_bytes(self) -> int:
        """KV、indexer key/scale 及整行 16 字节对齐所需的物理字节数。"""
        return (self.kv_bytes + self.indexer_dim + 4 + 15) // 16 * 16

    @property
    def indexer_offset(self) -> int:
        """保持既有格式：FP8 紧接 KV，浮点格式将对齐间隙放在 KV 与 indexer 之间。"""
        return self.kv_bytes if self.kv_encoding == "fp8_group128" else self.token_bytes - self.indexer_dim - 4

    def __post_init__(self) -> None:
        assert self.kv_encoding in ("fp16", "bf16", "fp8_group128"), "unknown packed MLA KV encoding"
        if self.kv_encoding == "fp8_group128":
            assert (
                self.dtype == torch.float8_e4m3fn
                and self.rope_dim == 64
                and self.latent_dim == 576
                and self.quant_group_size == 128
            ), "group128 packed MLA requires latent 576, rope 64 and group size 128"
        else:
            assert self.kv_encoding in ("fp16", "bf16")


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
        encoding = (
            f"fp8_group{self.config.quant_group_size}"
            if self.config.kv_encoding == "fp8_group128"
            else dtype_name(dtype=self.config.dtype)
        )
        return "mla_packed_" + encoding

    def allocate_storage(self, token_capacity: int) -> None:
        """按统一 token 容量分配本 State 的独立 KV 存储与 HOLD 页。"""
        self.capacity = token_capacity
        # FP8 行混合保存多种字段，用字节 buffer 承载，字段视图再解释各自的 dtype。
        storage_dtype = torch.uint8 if self.config.kv_encoding == "fp8_group128" else self.config.dtype
        self.storage = torch.empty(
            size=(
                len(self.config.logical_layer_groups),
                token_capacity + self.page_size,
                1,
                self.config.token_bytes // storage_dtype.itemsize,
            ),
            dtype=storage_dtype,
            device=self.device,
        )

    def get_token_bytes(self) -> int:
        """返回全部物理层的单 token 存储字节数，包含编码、indexer 和 padding。"""
        return len(self.config.logical_layer_groups) * self.config.token_bytes

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
        kv = (
            raw[..., : self.config.kv_bytes]
            if self.config.kv_encoding == "fp8_group128"
            else row[..., : self.config.latent_dim]
        )
        indexer = raw[..., self.config.indexer_offset : self.config.indexer_offset + self.config.indexer_dim + 4]
        return PackedMLALayerView(kv=kv, indexer_k=indexer)

    def write_full_att_kv(self, logical_layer_id: int, token_indices: torch.Tensor, kv: torch.Tensor) -> None:
        """写入 MLA KV 及其量化参数，保留同一 token 行中的 indexer 区域。"""
        assert token_indices.is_cuda, "token indices must reside on CUDA"
        storage_layer_index = self.get_storage_layer_index(logical_layer_id=logical_layer_id)
        if not len(token_indices):
            return
        assert self.storage.is_cuda, "KV writes require CUDA storage"
        if self.config.kv_encoding == "fp8_group128":
            from lightllm.common.basemodel.triton_kernel.kv_copy.destindex_copy_kv_flashmla_fp8 import (
                destindex_copy_kv_flashmla_fp8,
            )

            raw = self.storage[storage_layer_index]
            nope_dim = self.config.latent_dim - self.config.rope_dim
            scale_bytes = nope_dim // self.config.quant_group_size * torch.float32.itemsize
            rope_start = nope_dim + scale_bytes
            destindex_copy_kv_flashmla_fp8(
                KV_nope=kv[..., :nope_dim],
                KV_rope=kv[..., nope_dim:],
                DestLoc=token_indices,
                O_nope=raw[..., :nope_dim].view(torch.float8_e4m3fn),
                O_scale=raw[..., nope_dim:rope_start].view(torch.float32),
                O_rope=raw[..., rope_start : rope_start + self.config.rope_dim * torch.bfloat16.itemsize].view(
                    torch.bfloat16
                ),
            )
        else:
            from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

            destindex_copy_kv(
                K=kv, DestLoc=token_indices, Out=self.storage[storage_layer_index, ..., : self.config.latent_dim]
            )

    def copy_full_att_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        """复制完整 token 数据；索引使用 Tensor，源与目标位置由调用方保证不重叠。"""
        assert src_indices.is_cuda and dst_indices.is_cuda, "token indices must reside on CUDA"
        assert (
            src_indices.ndim == dst_indices.ndim == 1 and src_indices.numel() == dst_indices.numel()
        ), "token copies require equal one-dimensional index tensors"
        assert src_indices.dtype in (torch.int32, torch.int64) and dst_indices.dtype in (
            torch.int32,
            torch.int64,
        ), "token indices must be integer tensors"
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
    @cached_query
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
            kv_dim=self.config.latent_dim,
            indexer_offset=self.config.indexer_offset,
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
            kv_dim=self.config.latent_dim,
            indexer_offset=self.config.indexer_offset,
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
            remote_dp_full_att_state.config.kv_encoding,
            remote_dp_full_att_state.config.latent_dim,
            remote_dp_full_att_state.config.indexer_dim,
        ) == (
            self.config.kv_encoding,
            self.config.latent_dim,
            self.config.indexer_dim,
        ), "incompatible packed MLA format"
        assert (
            type(remote_dp_full_att_state) is type(self) and remote_dp_full_att_state.config == self.config
        ), "DP copy requires the same encoding and equivalent TP state configurations"
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

    @deepcopy_result
    @cached_query
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
            kv_dim=self.config.latent_dim,
            indexer_offset=self.config.indexer_offset,
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
            kv_dim=self.config.latent_dim,
            indexer_offset=self.config.indexer_offset,
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
        if config.kv_encoding == "fp8_group128":
            field_specs = [
                ("kv_nope", torch.uint8, 512, "dim"),
                ("kv_scale", torch.float32, 4, "group"),
                ("kv_rope", torch.bfloat16, 64, "dim"),
            ]
        else:
            field_specs = [("kv", config.dtype, config.latent_dim, "dim")]
        field_specs.extend(
            (
                ("indexer_k", torch.uint8, config.indexer_dim, "dim"),
                ("indexer_scale", torch.float32, 1, "scale"),
            )
        )
        for field_name, field_dtype, field_width, last_axis in field_specs:
            axes = ("layer", "token", "head", last_axis)
            local_shape = (len(config.logical_layer_groups), page_tokens, 1, field_width)
            global_shape = local_shape
            # 全部 TP ranks 保存同一 latent，只有 rank 0 写出共享外部页。
            write_enabled = config.tp_rank == 0
            axis_mappings = tuple(
                AxisMapping(
                    axis=axis,
                    local_start=0,
                    external_start=0,
                    length=dimension,
                )
                for axis, dimension in zip(axes, local_shape)
                if axis not in ("layer", "token")
            )

            # 阶段二：按字段 dtype 对齐起点，记录本字段区间，再推进下一个字段的字节偏移。
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
            "%s transfer regions built: fields=%d, total_bytes=%d",
            self.state_name,
            len(regions),
            byte_offset,
        )
        return tuple(regions)

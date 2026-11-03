from __future__ import annotations

from lightllm.common.attention_state.utils import weak_cached_method
from typing import TYPE_CHECKING, List, Literal
from dataclasses import dataclass
from math import prod

import torch

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

from ...utils import deepcopy_result, dtype_name
from ...types import AxisMapping, TransferRegion, TensorField
from ..base_state import FullAttentionState


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

    storage_fields = ("storage",)

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

    def get_storage_specs(self, token_capacity):
        from ...storage import TensorStorageSpec

        dtype = torch.uint8 if self.config.kv_encoding == "fp8_group128" else self.config.dtype
        shape = (
            len(self.config.logical_layer_groups),
            token_capacity + self.page_size,
            1,
            self.config.token_bytes // dtype.itemsize,
        )
        return (TensorStorageSpec("storage", shape, dtype),)

    def get_transfer_fields(self):
        raw = self.storage.view(torch.uint8)
        if self.config.kv_encoding == "fp8_group128":
            fields = {
                "kv_nope": (raw[..., :512], "dim"),
                "kv_scale": (raw[..., 512:528].view(torch.float32), "group"),
                "kv_rope": (raw[..., 528:656].view(torch.bfloat16), "dim"),
            }
        else:
            fields = {"kv": (self.storage[..., : self.config.latent_dim], "dim")}
        start = self.config.indexer_offset
        end = start + self.config.indexer_dim
        fields["indexer_k"] = raw[..., start:end], "dim"
        fields["indexer_scale"] = raw[..., end : end + 4].view(torch.float32), "scale"
        return {name: TensorField(tensor, ("layer", "token", "head", axis)) for name, (tensor, axis) in fields.items()}

    def get_token_copy_fields(self):
        return {"packed": TensorField(self.storage, ("layer", "token", "head", "byte"))}

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

    # ========================================================================
    # 三、CPU Cache 数据交换
    # ========================================================================

    @deepcopy_result
    @weak_cached_method
    def get_cpu_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 CPU 页大小返回字段区间与 TP 切片，作为 CPU 页容量计算依据。

        覆盖 KV、动态量化参数及 indexer 有效字段，不包含 GPU 行 padding 或静态标定参数。
        """
        page_tokens = get_env_start_args().cpu_cache_token_page_size
        return self._get_transfer_regions(page_tokens=page_tokens)

    # ========================================================================
    # 四、DP 迁移
    # ========================================================================

    # ========================================================================
    # 五、PD 分离传输
    # ========================================================================

    @deepcopy_result
    @weak_cached_method
    def get_pd_transfer_regions(self) -> tuple[TransferRegion, ...]:
        """按全局 PD 页大小返回字段与切片，交由管理器安排其在联合页中的位置。"""
        page_tokens = get_env_start_args().pd_kv_page_size
        return self._get_transfer_regions(page_tokens=page_tokens)

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

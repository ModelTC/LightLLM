from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, ClassVar
from math import lcm
import torch
from ...types import FieldName
from ..token_storage import TokenField, TokenStorageSpec, PackedField, PackedTokenStorageSpec
from ...utils import dtype_name


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

    @cached_property
    def storage_spec(self):
        return packed_mla_spec(self)

    @property
    def kv_bytes(self) -> int:
        field = self.storage_spec.field(FieldName.KV_DATA)
        return field.width * field.dtype.itemsize

    @property
    def token_bytes(self) -> int:
        field = self.storage_spec.fields[0]
        return field.width * field.dtype.itemsize

    @property
    def indexer_offset(self) -> int:
        return self.storage_spec.view(FieldName.INDEXER).byte_offset

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
    """借用同一 packed 存储的计算字段；冻结属性不冻结 Tensor 内容。"""

    kv: torch.Tensor
    indexer_k: torch.Tensor


@dataclass(frozen=True)
class FP8E8M0PackedMLAConfig:
    """固定 group64/E8M0 编码，只记录调用方能够选择的原子布局信息。"""

    # 外层为物理存储层，内层为共享该层全部 KV/indexer 的逻辑层号。
    logical_layer_groups: tuple[tuple[int, ...], ...]
    # 每个 token 的 indexer key 维度；其后另保存一个 FP32 scale，不包含在该维度中。
    indexer_dim: int = 128
    # 完整 latent 在各 TP rank 复制；rank 0 写出共享外部页，各 rank 均可读回。
    tp_rank: int = 0
    # 所属 TP 组大小，不增加物理 KV head，也不改变单 token 的存储字节数。
    tp_world_size: int = 1

    # 算子输入的完整 latent 维度，不是量化后的字节数，也不包含 indexer。
    LATENT_DIM: ClassVar[int] = 512
    # 固定编码：448 个 FP8 元素 + 64 个 BF16 元素 + 7 个 E8M0 scale + 1 字节对齐。
    # 末尾 64 维保留 BF16 不代表启用了 RoPE，GLM 使用 NoPE attention。
    KV_BYTES: ClassVar[int] = 584

    @cached_property
    def storage_spec(self):
        return fp8_e8m0_spec(self)

    @property
    def payload_bytes(self) -> int:
        return self.storage_spec.field(FieldName.KV).width

    @property
    def token_bytes(self) -> int:
        return self.storage_spec.fields[0].width

    def __post_init__(self) -> None:
        assert self.logical_layer_groups and all(self.logical_layer_groups)
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert len(layers) == len(set(layers)), "logical layers cannot repeat across storage groups"
        assert self.indexer_dim > 0 and 0 <= self.tp_rank < self.tp_world_size


def packed_mla_spec(config):
    """一次定义物理行、计算切片和线上字段；页面只序列化选中的有效字段。"""
    fp8 = config.kv_encoding == "fp8_group128"
    dtype = torch.uint8 if fp8 else config.dtype
    if fp8:
        nope = config.latent_dim - config.rope_dim
        scales = nope // config.quant_group_size
        rope_offset = nope + scales * torch.float32.itemsize
        kv_bytes = rope_offset + config.rope_dim * torch.bfloat16.itemsize
        payload = (
            PackedField(TokenField(FieldName.KV_NOPE, torch.uint8, nope), 0),
            PackedField(TokenField(FieldName.KV_SCALE, torch.float32, scales, "group"), nope),
            PackedField(TokenField(FieldName.KV_ROPE, torch.bfloat16, config.rope_dim), rope_offset),
        )
    else:
        kv_bytes = config.latent_dim * dtype.itemsize
        payload = (PackedField(TokenField(FieldName.KV, dtype, config.latent_dim), 0),)
    indexer_bytes = config.indexer_dim + torch.float32.itemsize
    row_bytes = (kv_bytes + indexer_bytes + 15) // 16 * 16
    # 旧浮点格式把 padding 放在 KV 与 indexer 之间；FP8 放在行尾，保持线上格式。
    indexer_offset = kv_bytes if fp8 else row_bytes - indexer_bytes
    payload += (
        PackedField(TokenField(FieldName.INDEXER_K, torch.uint8, config.indexer_dim), indexer_offset),
        PackedField(
            TokenField(FieldName.INDEXER_SCALE, torch.float32, 1, "scale"), indexer_offset + config.indexer_dim
        ),
    )
    return PackedTokenStorageSpec(
        encoding="mla_packed_" + (f"fp8_group{config.quant_group_size}" if fp8 else dtype_name(config.dtype)),
        logical_layer_groups=config.logical_layer_groups,
        global_head_num=1,
        head_num=1,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        fields=(TokenField(FieldName.KV, dtype, row_bytes // dtype.itemsize),),
        views=payload
        + (
            PackedField(TokenField(FieldName.KV_DATA, dtype, kv_bytes // dtype.itemsize), 0),
            PackedField(TokenField(FieldName.INDEXER, torch.uint8, indexer_bytes), indexer_offset),
        ),
        page_fields=tuple(part.field.name for part in payload),
    )


def fp8_e8m0_spec(config):
    indexer_bytes = config.indexer_dim + torch.float32.itemsize
    payload_bytes = config.KV_BYTES + indexer_bytes
    row_bytes = (payload_bytes + 15) // 16 * 16
    return PackedTokenStorageSpec(
        encoding="mla_packed_fp8_group64_e8m0",
        logical_layer_groups=config.logical_layer_groups,
        global_head_num=1,
        head_num=1,
        tp_rank=config.tp_rank,
        tp_world_size=config.tp_world_size,
        fields=(TokenField(FieldName.KV, torch.uint8, row_bytes),),
        views=(
            PackedField(TokenField(FieldName.KV, torch.uint8, payload_bytes), 0),
            PackedField(TokenField(FieldName.KV_DATA, torch.uint8, config.KV_BYTES), 0),
            PackedField(TokenField(FieldName.INDEXER, torch.uint8, indexer_bytes), config.KV_BYTES),
        ),
        page_fields=(FieldName.KV,),
    )

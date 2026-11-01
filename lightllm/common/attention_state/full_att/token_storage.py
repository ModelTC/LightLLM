from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import prod

import torch

from ..types import AxisMapping, FieldName, TensorField, TransferRegion


@dataclass(frozen=True)
class TokenField:
    """字段的编码元素宽度及轴语义；width 的单位是 dtype 元素，不是逻辑 head_dim。"""

    name: str
    dtype: torch.dtype
    width: int
    width_axis: str = "dim"

    def __post_init__(self):
        assert self.name and self.width > 0
        assert self.width_axis not in ("layer", "token", "head")

    @property
    def axes(self):
        return ("layer", "token", "head", self.width_axis)


@dataclass(frozen=True)
class TokenStorageSpec:
    """不可变的本地物理布局，同时派生容量、计算视图及全局 CPU/PD 区域。

    logical_layer_groups 的每组只分配一层；block_tokens 仍以原始 token 为单位。
    编码和形状相同不代表模型或静态标定相同，复用者仍须验证这些计算身份。
    """

    encoding: str
    logical_layer_groups: tuple[tuple[int, ...], ...]
    global_head_num: int
    head_num: int
    tp_rank: int
    tp_world_size: int
    fields: tuple[TokenField, ...]
    block_tokens: int = 1
    zero_initialize: bool = False
    format_key: tuple[int, ...] = ()

    def __post_init__(self):
        assert self.block_tokens > 0
        assert self.fields and len({field.name for field in self.fields}) == len(self.fields)
        assert self.logical_layer_groups and all(self.logical_layer_groups)
        layers = [layer for group in self.logical_layer_groups for layer in group]
        assert len(layers) == len(set(layers)) and all(layer >= 0 for layer in layers)
        assert self.head_num > 0 and self.global_head_num > 0
        assert self.global_head_num % self.head_num == 0, "head fields require complete shards or replicas"
        assert 0 <= self.tp_rank < self.tp_world_size
        assert (self.head_num * self.tp_world_size) % self.global_head_num == 0

    @property
    def token_bytes(self) -> int:
        return (
            len(self.logical_layer_groups)
            * self.head_num
            * sum(field.width * field.dtype.itemsize for field in self.fields)
        )

    def shape(self, field: TokenField, token_slots: int) -> tuple[int, ...]:
        return (len(self.logical_layer_groups), token_slots, self.head_num, field.width)

    @property
    def transfer_fields(self) -> tuple[TokenField, ...]:
        return self.fields

    @cached_property
    def _fields_by_name(self):
        return {field.name: field for field in self.fields}

    def field(self, name: FieldName) -> TokenField:
        return self._fields_by_name[name]

    def tensor(self, buffers, name: FieldName) -> torch.Tensor:
        return buffers[name]

    def page_views(self, buffers) -> dict[FieldName, TensorField]:
        return {field.name: TensorField(self.tensor(buffers, field.name), field.axes) for field in self.transfer_fields}

    def transfer_regions(self, page_tokens: int) -> tuple[TransferRegion, ...]:
        assert (
            page_tokens > 0 and page_tokens % self.block_tokens == 0
        ), "CPU/PD page token capacity must be a positive multiple of block_token_size"
        replicas = self.tp_world_size * self.head_num // self.global_head_num
        head_start = self.tp_rank // replicas * self.head_num
        regions, offset = [], 0
        for field in self.transfer_fields:
            shape = (len(self.logical_layer_groups), page_tokens, self.global_head_num, field.width)
            offset = (offset + field.dtype.itemsize - 1) // field.dtype.itemsize * field.dtype.itemsize
            end = offset + prod(shape) * field.dtype.itemsize
            regions.append(
                TransferRegion(
                    field=field.name.value if isinstance(field.name, FieldName) else field.name,
                    byte_start=offset,
                    byte_end=end,
                    dtype=field.dtype,
                    axes=field.axes,
                    shape=shape,
                    axis_mappings=(
                        AxisMapping("head", 0, head_start, self.head_num),
                        AxisMapping(field.width_axis, 0, 0, field.width),
                    ),
                    write_enabled=self.tp_rank % replicas == 0,
                )
            )
            offset = end
        return tuple(regions)


@dataclass(frozen=True)
class PackedField:
    """同一物理行（或连续多行）中的借用视图，偏移以字节计。

    多个视图允许重叠，例如整个 KV 和其 NoPE/scale/RoPE 子字段。
    block_tokens > 1 时覆盖完整压缩块，只用于计算，不作为逐 token 传输字段。
    """

    field: TokenField
    byte_offset: int
    block_tokens: int = 1


@dataclass(frozen=True)
class PackedTokenStorageSpec(TokenStorageSpec):
    # views 是唯一的切片定义；page_fields 只选择传输顺序，不再次定义偏移、dtype 或轴。
    views: tuple[PackedField, ...] = ()
    page_fields: tuple[FieldName, ...] = ()

    def __post_init__(self):
        super().__post_init__()
        assert len(self.fields) == 1 and self.views and self.page_fields
        row_bytes = self.fields[0].width * self.fields[0].dtype.itemsize
        assert len({part.field.name for part in self.views}) == len(self.views)
        assert len(set(self.page_fields)) == len(self.page_fields)
        for part in self.views:
            assert part.block_tokens > 0 and self.block_tokens % part.block_tokens == 0
            assert part.block_tokens == 1 or self.head_num == 1
            assert part.byte_offset >= 0 and part.byte_offset % part.field.dtype.itemsize == 0
            assert part.byte_offset + part.field.width * part.field.dtype.itemsize <= row_bytes * part.block_tokens
        for name in self.page_fields:
            assert self.view(name).block_tokens == 1, "page fields must use original token slots"

    @cached_property
    def _views_by_name(self):
        return {part.field.name: part for part in self.views}

    def view(self, name: FieldName) -> PackedField:
        return self._views_by_name[name]

    def field(self, name: FieldName) -> TokenField:
        return self.view(name).field

    @property
    def transfer_fields(self):
        return tuple(self.field(name) for name in self.page_fields)

    def tensor(self, buffers, name: FieldName):
        part = self.view(name)
        raw = buffers[self.fields[0].name].view(torch.uint8)
        if part.block_tokens != 1:
            raw = raw.view(raw.shape[0], -1, 1, raw.shape[-1] * part.block_tokens)
        return raw[..., part.byte_offset : part.byte_offset + part.field.width * part.field.dtype.itemsize].view(
            part.field.dtype
        )


class TokenStorage:
    """拥有物理 Tensor，不拥有 token 槽位分配权或缓存有效性。

    计算和传输的视图都借用这些 Tensor；IPC 重绑后按当前 buffer 重新取视图。
    调用方须在异步使用结束后才能释放槽位、替换 buffer 或销毁 storage。
    """

    def __init__(self, spec: TokenStorageSpec):
        self.spec = spec
        self.capacity = 0
        self.buffers: dict[str, torch.Tensor] = {}
        self._layer_indices = {
            logical: physical for physical, group in enumerate(spec.logical_layer_groups) for logical in group
        }

    def layer_view(self, field: FieldName, logical_layer_id: int) -> torch.Tensor:
        """从规格取字段并选择物理层，保留 packed stride，不分配 payload 副本。"""
        return self.spec.tensor(self.buffers, field)[self.layer_index(logical_layer_id)]

    def layer_index(self, logical_layer_id: int) -> int:
        return self._layer_indices[logical_layer_id]

    def contains_layer(self, logical_layer_id: int) -> bool:
        return logical_layer_id in self._layer_indices

    def allocate(self, token_capacity: int, hold_tokens: int, device: torch.device) -> None:
        """分配正常容量和 HOLD 尾部；分配成功不表示任意槽位已写入有效 KV。"""
        assert token_capacity >= 0 and hold_tokens > 0
        assert (
            hold_tokens % self.spec.block_tokens == 0
        ), "allocator page_size must be a positive multiple of block_token_size"
        assert token_capacity % self.spec.block_tokens == 0, "token capacity must contain complete compressed blocks"
        allocate = torch.zeros if self.spec.zero_initialize else torch.empty
        buffers = {
            field.name: allocate(
                self.spec.shape(field, token_capacity + hold_tokens),
                dtype=field.dtype,
                device=device,
            )
            for field in self.spec.fields
        }
        self.buffers = buffers
        self.capacity = token_capacity

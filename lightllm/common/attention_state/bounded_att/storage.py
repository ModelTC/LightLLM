from dataclasses import dataclass

import torch

from ..types import FieldName, TensorField, TransferRegion


@dataclass(frozen=True)
class RuntimeBuffer:
    """请求运行态的完整物理形状，包含该表示需要的 layer/request/MTP 轴。"""

    name: FieldName
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class SnapshotField:
    """快照字段及其页池属性；axes 不包含已选定的 page 槽位轴。"""

    name: FieldName
    attribute: str
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class BoundedStorageSpec:
    """运行态、快照及 CPU/PD 区域的不可变规格。

    CPU 快照和 PD 继续计算窗口可不同；例如 compressor 在步长边界只需保存重叠历史。
    """

    encoding: str
    logical_layers: tuple[int, ...]
    runtime_buffers: tuple[RuntimeBuffer, ...]
    snapshot_fields: tuple[SnapshotField, ...]
    cpu_regions: tuple[TransferRegion, ...]
    pd_regions: tuple[TransferRegion, ...]


class BoundedStorage:
    """拥有请求 Tensor 与快照页池，页池槽位由 manager 的 allocator 分配。

    构造即分配固定运行态，页池容量稍后确定。分配不建立前缀身份，
    也不证明某一槽位已写入；发布与异步使用结束由上层编排。
    """

    def __init__(self, spec, request_capacity, mtp_step, device, page_pool_factory):
        self.spec = spec
        self.request_capacity = request_capacity
        self.mtp_step = mtp_step
        self.layer_indices = {layer: index for index, layer in enumerate(spec.logical_layers)}
        self.buffers = {
            field.name: torch.zeros(field.shape, dtype=field.dtype, device=device) for field in spec.runtime_buffers
        }
        self.page_pool_factory = page_pool_factory
        self.big_pages_pool = None
        self.small_pages_pool = None

    def tensor(self, field: FieldName):
        return self.buffers[field]

    def initialize_page_pools(self, big_page_capacity, small_page_capacity=0, device="cpu", pin_memory=True):
        """分别构造两个页池后一次绑定；页池不分配或回收逻辑槽位。"""
        assert self.big_pages_pool is None and self.small_pages_pool is None, "page pools already initialized"
        assert big_page_capacity >= 0 and small_page_capacity >= 0, "page capacities must be nonnegative"
        if self.page_pool_factory is None:
            return
        pools = [
            self.page_pool_factory(size=capacity, device=device, pin_memory=pin_memory) if capacity else None
            for capacity in (big_page_capacity, small_page_capacity)
        ]
        self.big_pages_pool, self.small_pages_pool = pools

    def pool(self, page_kind):
        assert page_kind in ("big", "small"), f"unknown page kind {page_kind}"
        pool = self.big_pages_pool if page_kind == "big" else self.small_pages_pool
        assert pool is not None, f"{page_kind} pages are disabled"
        return pool

    def bind_field(self, name: FieldName, tensor: torch.Tensor) -> TensorField:
        """按规格绑定已选定请求/页的视图，避免计算侧再次声明字段轴。"""
        region = next(region for region in self.spec.pd_regions if region.field == name)
        return TensorField(tensor, region.axes)

    def page_fields(self, pool, slot):
        """借用所选页池槽位的字段，调用方须保持该槽位存活直到搬运完成。"""
        return {
            field.name: TensorField(getattr(pool, field.attribute)[slot], field.axes)
            for field in self.spec.snapshot_fields
        }

"""Bounded attention 的 CPU/PD 页传输：复制一份完整状态快照。"""

from collections.abc import Mapping

import torch

from ..types import StateTransferLayout, TensorField, TransferRegion


# ========================================================================
# 一、CPU/PD 页交换：按状态字段生成复制视图
# ========================================================================

def bounded_att_buffer_copies(
    fields: Mapping[str, TensorField],
    page_buffers: torch.Tensor,
    layout: StateTransferLayout,
    page_index: int,
    write: bool,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """生成一份 bounded 状态与外部页之间的复制计划，不处理 token 数量。

    fields 由 State 选取运行态或大小页快照，例如 KDA/GDN 的 Conv、SSM。
    每份快照完整复制；Conv 的 Q/K/V 区间可以分别映射到同一全局字段。
    page_buffers 至少为二维，第一维统一表示页；不接受没有页维度的单页 Tensor。
    page_index 必须显式传入；CPU/PD 调用方均传入页集合和实际物理页索引。
    返回 (源视图, 目标视图) 列表，由调用方遍历执行 dst.copy_(src, non_blocking=True)。
    """
    # CPU/PD 统一沿第一维选页，再按字节解释页内存储，不复制数据。
    assert page_buffers.ndim >= 2, "page buffers must include a leading page dimension"
    page_buffer = page_buffers[page_index]
    page_bytes = page_buffer.view(torch.uint8).view(-1)
    copies = []
    for region in layout.regions:
        # 阶段一：定位快照字段和外部页中的全局状态区域。
        assert region.field in fields, f"unknown transfer field {region.field}"
        field = fields[region.field]
        # region 起止位置相对整页，按字段 dtype 和 shape 直接构造外部视图。
        external = page_bytes[region.byte_start:region.byte_end].view(region.dtype).view(region.shape)
        if write and not region.write_enabled:
            continue

        # 阶段二：按 channel/head 等 TP 切片生成视图，Conv 窗口及 SSM 状态完整保留。
        local_view, external_view = _copy_pair(field, external, region)
        # 比较切片后的实际复制范围，避免 copy_ 隐式广播掩盖布局错误。
        assert local_view.shape == external_view.shape, "local and external copy views must have the same shape"
        # 写出只由选定的 TP 副本执行；读回覆盖每个副本自己的分片。
        copies.append((local_view, external_view) if write else (external_view, local_view))
    return copies


# ========================================================================
# 二、私有辅助：本类状态的复制视图
# ========================================================================

def _copy_pair(
    field: TensorField,
    external: torch.Tensor,
    region: TransferRegion,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按 bounded 状态的 channel/head 等映射取视图，不处理 token 裁剪。"""
    assert field.tensor.dtype == external.dtype, "local and external field dtypes must match"
    # 外部布局可采用不同轴顺序，先转换为本地字段的顺序；只创建视图。
    external = external.permute(tuple(region.axes.index(axis) for axis in field.axes))
    local_slices = [slice(None)] * field.tensor.ndim
    external_slices = [slice(None)] * external.ndim
    # 布局已明确每个分片的两侧起点和长度，其余轴完整保留。
    for axis_slice in region.axis_slices:
        axis = field.axes.index(axis_slice.axis)
        local_slices[axis] = slice(axis_slice.src_start, axis_slice.src_start + axis_slice.length)
        external_slices[axis] = slice(axis_slice.dst_start, axis_slice.dst_start + axis_slice.length)
    return field.tensor[tuple(local_slices)], external[tuple(external_slices)]

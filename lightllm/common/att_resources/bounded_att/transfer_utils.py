"""Bounded attention 的 CPU/PD 页传输：复制一份完整状态快照。"""

from collections.abc import Mapping

import torch

from ..types import StateTransferLayout, TensorField, TransferRegion


# ========================================================================
# 一、CPU/PD 页交换：按状态字段生成复制视图
# ========================================================================

def bounded_att_buffer_copies(
    page_fields: list[Mapping[str, TensorField]],
    page_buffers: torch.Tensor,
    layout: StateTransferLayout,
    page_indices: list[int],
    write: bool,
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """按传输区域收集所有页面的复制视图，每个返回批次只包含一种 shape。

    page_fields 与 page_indices 一一对应，调用方已排除 ready 页。
    Conv 的 Q/K/V 切片与 SSM 分别成批，不将不同字段或不同 TP 切片混入同一次拷贝。
    page_buffers 的第一维为物理页；这里只创建视图，由调用方逐批提交 Triton 算子。
    """
    assert page_buffers.ndim >= 2, "page buffers must include a leading page dimension"
    batches = []
    for region in layout.regions:
        if write and not region.write_enabled:
            continue
        copies = []
        for fields, page_index in zip(page_fields, page_indices):
            # 阶段一：按字节区间定位当前页的字段，构造外部存储视图。
            field = fields[region.field]
            page_bytes = page_buffers[page_index].view(torch.uint8).view(-1)
            external = page_bytes[region.byte_start:region.byte_end].view(region.dtype).view(region.shape)
            # 阶段二：同一区域使用相同的 TP 切片，所有页面生成的视图 shape 一致。
            local_view, external_view = _copy_pair(field, external, region)
            assert local_view.shape == external_view.shape, "local and external copy views must have the same shape"
            copies.append((local_view, external_view) if write else (external_view, local_view))
        batches.append(copies)
    return batches


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

"""Bounded attention 的 CPU/PD 页传输：复制一份完整状态快照。"""

from collections.abc import Mapping

import torch

from ..types import StateTransferLayout, TensorField


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
    CPU 路径的 page_fields 来自本地大小页池，PD 路径来自请求运行态；均为本地 TP 分片。
    layout.regions 描述外部页中的全局字段，以及本地 TP 分片在该字段中的位置。
    axis_slices 的 src_start 固定指本地，dst_start 固定指外部页，读回时也不交换含义。
    CPU 页和大小页快照的窗口均按逻辑时间排列，此处只处理普通轴切片，不展开或恢复 ring。
    Conv 的 Q/K/V 切片与 SSM 分别成批，不将不同字段或不同 TP 切片混入同一次拷贝。
    page_buffers 的第一维为物理页；这里只创建视图，由调用方逐批提交 Triton 算子。
    """
    assert page_buffers.ndim >= 2, "page buffers must include a leading page dimension"
    batches = []
    for region in layout.regions:
        if write and not region.write_enabled:
            continue
        copies = []
        for local_fields, page_index in zip(page_fields, page_indices):
            # 阶段一：region.field 选择本地 Conv/SSM；CPU 读回时这里就是目标大页槽位。
            local_field = local_fields[region.field]
            local_tensor = local_field.tensor

            # 阶段二：region 的字节区间和 shape 定位外部页中的全局字段，再对齐轴顺序。
            page_bytes = page_buffers[page_index].view(torch.uint8).view(-1)
            external_tensor = page_bytes[region.byte_start:region.byte_end].view(region.dtype).view(region.shape)
            assert local_tensor.dtype == external_tensor.dtype, "local and external field dtypes must match"
            external_tensor = external_tensor.permute(tuple(region.axes.index(axis) for axis in local_field.axes))

            # 阶段三：region 的两侧切片定位同一个 TP 分片，不代表本次复制的源和目标。
            # 例如 rank 1 的本地 head[0:4] 对应外部页 head[4:8]，写出和读回使用相同映射。
            local_slices = [slice(None)] * local_tensor.ndim
            external_slices = [slice(None)] * external_tensor.ndim
            for axis_slice in region.axis_slices:
                axis = local_field.axes.index(axis_slice.axis)
                local_slices[axis] = slice(axis_slice.src_start, axis_slice.src_start + axis_slice.length)
                external_slices[axis] = slice(axis_slice.dst_start, axis_slice.dst_start + axis_slice.length)
            local_view = local_tensor[tuple(local_slices)]
            external_view = external_tensor[tuple(external_slices)]
            assert local_view.shape == external_view.shape, "local and external copy views must have the same shape"

            # 阶段四：只在此确定搬运方向；CPU 读回为外部页 → 本地大页池，不直接恢复请求。
            if write:
                copies.append((local_view, external_view))
            else:
                copies.append((external_view, local_view))
        batches.append(copies)
    return batches

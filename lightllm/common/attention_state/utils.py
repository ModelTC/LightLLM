"""注意力 State 的命名和传输视图工具。"""

from __future__ import annotations

import torch

MAX_DRAFT_MODELS = 36
MODEL_ROLES = ("main", *(f"draft{i}" for i in range(MAX_DRAFT_MODELS)))


# ========================================================================
# 一、State 命名与成员管理
# ========================================================================

def dtype_name(dtype: torch.dtype):
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dtype]


# ========================================================================
# 二、传输视图：合并两端共同支持的尾部维度
# ========================================================================


def merge_contiguous_tail_dims(
    copies: list[tuple[torch.Tensor, torch.Tensor]],
    start_dim: int = 0,
    keepdim: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """合并批次两侧共同连续的尾部维度，只创建 view，不复制数据或改变 dtype。

    调用方保证批次及 Tensor 非空，同一侧的 shape/stride 相同，两侧待合并的尾部 shape 相同；
    前缀维度的数量和容量可以不同，例如 GPU token 存储与外部页集合。
    根据首对 Tensor 确定共同的合并范围，再为整批分别创建两侧形状的视图。
    从末轴向前尝试，遇到任一侧无法合并就停止，不单独整理前面的维度。

    start_dim 是允许参与合并的最前轴，之前的轴保持独立。例如
    [layer, window, head, dim] 传入 start_dim=2，只合并 head/dim，保留环形 window。
    支持负数，例如 start_dim=-2 对齐两侧最后两维，不要求两侧前缀维数一致。
    keepdim=True 将被合并的前面各轴保留为 1，元素数集中到末轴，供固定维数的
    token/page kernel 使用；默认 False 直接减少维数，供 bounded 聚合 kernel 使用。
    直接按 shape/stride 判断连续性，长度为 1 的轴不约束 stride。
    只依赖公开 Tensor 接口，确定形状后一次性创建 view，不使用可能复制数据的 reshape。
    """
    source, destination = copies[0]
    source_shape, destination_shape = source.shape, destination.shape
    source_strides, destination_strides = source.stride(), destination.stride()
    ndim = len(source_shape)
    assert -ndim <= start_dim < ndim, "invalid tail merge boundary"
    if start_dim < 0:
        start_dim += ndim
    tail_dims = ndim - start_dim
    assert len(destination_shape) >= tail_dims and source_shape[start_dim:] == destination_shape[-tail_dims:], (
        "copy tail shapes must match"
    )

    # 阶段一：从末轴向前检查；两侧的前缀容量和轴数独立，按距末轴的距离对齐。
    first_tail_dim = ndim - 1
    tail_elements = source_shape[-1]
    source_tail_stride, destination_tail_stride = source_strides[-1], destination_strides[-1]
    prefix_dim_difference = len(destination_shape) - ndim
    for source_axis in range(ndim - 2, start_dim - 1, -1):
        destination_axis = source_axis + prefix_dim_difference
        axis_size = source_shape[source_axis]
        if axis_size != 1:
            if tail_elements == 1:
                # 内侧只有一个元素，其 stride 不影响寻址；以当前非单元素轴为准。
                source_tail_stride = source_strides[source_axis]
                destination_tail_stride = destination_strides[destination_axis]
            elif (
                source_strides[source_axis] != tail_elements * source_tail_stride
                or destination_strides[destination_axis] != tail_elements * destination_tail_stride
            ):
                break
        # 当前轴长度为 1 时忽略其 stride，但保留已累计的连续区间和元素步长。
        tail_elements *= axis_size
        first_tail_dim = source_axis

    if first_tail_dim == ndim - 1:
        return copies

    # 阶段二：整批复用普通 tuple 形状，减少 view 的参数解析开销。
    # keepdim 保留固定维数 kernel 所需的单元素轴。
    merge_dims = ndim - first_tail_dim
    tail_shape = (1,) * (merge_dims - 1) + (tail_elements,) if keepdim else (tail_elements,)
    source_transfer_shape = tuple(source_shape[:-merge_dims]) + tail_shape
    destination_transfer_shape = tuple(destination_shape[:-merge_dims]) + tail_shape
    if source_transfer_shape == source_shape and destination_transfer_shape == destination_shape:
        return copies
    return [(src.view(source_transfer_shape), dst.view(destination_transfer_shape)) for src, dst in copies]


# ========================================================================
# 三、传输视图：为整批选择最宽的搬运 dtype
# ========================================================================


def view_max_copy_dtype(
    copies: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """将整批视图按最大可用整数宽度搬运，不转换数值、不复制数据。

    调用方保证批次非空、dtype 相同，同一侧的 shape/stride 相同，两侧的末轴长度相同。
    两侧的外层 shape/stride 可以不同。依次尝试 8/4/2 字节宽度，整批统一选择一种 dtype；
    不能扩大时返回原列表。只改变末轴的元素数，保留其余轴及存储间隙。
    本函数不合并维度；有需要时先调用 merge_contiguous_tail_dims。
    先用末轴字节数、首对 Tensor 的外层 stride 和 Storage 偏移过滤候选；
    同一侧的 stride 由整批共用，其余切片的 Storage 偏移继续由 view 校验。
    只捕获创建视图的 RuntimeError，任一视图失败就整批退回较窄的 dtype。
    """
    source, destination = copies[0]
    element_bytes = source.element_size()
    if element_bytes >= 8 or source.stride(-1) != 1 or destination.stride(-1) != 1:
        return copies
    row_bytes = source.shape[-1] * element_bytes
    # 按行宽选择候选类型，例如 12 字节只尝试 INT32、INT16。
    if row_bytes % 8 == 0:
        copy_dtypes = ((8, torch.int64), (4, torch.int32), (2, torch.int16))
    elif row_bytes % 4 == 0:
        copy_dtypes = ((4, torch.int32), (2, torch.int16))
    elif row_bytes % 2 == 0:
        copy_dtypes = ((2, torch.int16),)
    else:
        return copies

    # 首对布局即可排除必然失败的候选，避免 MTP Conv 等带间隙切片反复抛异常。
    # 这些值以原 dtype 的元素数计量，扩宽后必须仍能表示为整数 stride/offset。
    strides_and_offsets = (
        *source.stride()[:-1],
        *destination.stride()[:-1],
        source.storage_offset(),
        destination.storage_offset(),
    )
    for transfer_bytes, transfer_dtype in copy_dtypes:
        if transfer_bytes <= element_bytes:
            break
        elements_per_unit = transfer_bytes // element_bytes
        if any(value % elements_per_unit for value in strides_and_offsets):
            continue
        # 批次中其他切片的起点可能不同；保留 view 的整批校验和较窄类型回退。
        try:
            return [(src.view(transfer_dtype), dst.view(transfer_dtype)) for src, dst in copies]
        except RuntimeError:
            continue
    return copies

"""注意力 State 的命名和传输视图工具。"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch._prims import _collapse_view_helper

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

    调用方保证同一侧的 shape/stride 相同，两侧待合并的尾部 shape 相同；
    前缀维度的数量和容量可以不同，例如 GPU token 存储与外部页集合。
    根据首对 Tensor 确定共同的合并范围，再为整批分别创建两侧形状的视图。
    从末轴向前尝试，遇到任一侧无法合并就停止，不单独整理前面的维度。

    start_dim 是允许参与合并的最前轴，之前的轴保持独立。例如
    [layer, window, head, dim] 传入 start_dim=2，只合并 head/dim，保留环形 window。
    支持负数，例如 start_dim=-2 对齐两侧最后两维，不要求两侧前缀维数一致。
    keepdim=True 将被合并的前面各轴保留为 1，元素数集中到末轴，供固定维数的
    token/page kernel 使用；默认 False 直接减少维数，供 bounded 聚合 kernel 使用。
    长度为 1 的轴是否可以合并由 PyTorch 根据布局判断。
    通过版本断言限制为已验证的 PyTorch 2.11.x，end 包含在合并区间内。
    helper 对带 padding 的单元素轴可能过于宽松，因此最终以 view 是否成功为准；
    只捕获创建视图的 RuntimeError，退回较小的合并范围，不使用可能复制数据的 reshape。
    """
    _check_collapse_view_version(torch_version=torch.__version__)
    source, destination = copies[0]
    assert -source.ndim <= start_dim < source.ndim, "invalid tail merge boundary"
    if start_dim < 0:
        start_dim += source.ndim
    tail_dims = source.ndim - start_dim
    assert destination.ndim >= tail_dims and source.shape[start_dim:] == destination.shape[-tail_dims:], (
        "copy tail shapes must match"
    )
    transfer_shapes = []
    last_dim = source.ndim - 1
    for first_dim in range(last_dim - 1, start_dim - 1, -1):
        # 按距末轴的距离对齐区间；两侧独立保留 layer/token/page 等前缀。
        merge_dims = source.ndim - first_dim
        destination_first_dim = destination.ndim - merge_dims
        source_shape, _ = _collapse_view_helper(
            a=source,
            start=first_dim,
            end=last_dim,
            must_be_valid=None,
        )
        if source_shape is None:
            break
        destination_shape, _ = _collapse_view_helper(
            a=destination,
            start=destination_first_dim,
            end=destination.ndim - 1,
            must_be_valid=None,
        )
        if destination_shape is None:
            break
        if keepdim:
            singleton_axes = (1,) * (merge_dims - 1)
            source_shape = source.shape[:first_dim] + singleton_axes + (source_shape[-1],)
            destination_shape = destination.shape[:destination_first_dim] + singleton_axes + (destination_shape[-1],)
        if source_shape != source.shape or destination_shape != destination.shape:
            transfer_shapes.append((source_shape, destination_shape))

    # 优先采用合并范围最大的形状。普通布局只创建一次整批视图；
    # helper 误判时逐步缩小合并范围，保留跨轴的存储间隙。
    for source_shape, destination_shape in reversed(transfer_shapes):
        try:
            return [(src.view(source_shape), dst.view(destination_shape)) for src, dst in copies]
        except RuntimeError:
            continue
    return copies


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
    先用末轴字节数缩小候选范围，再由 view 校验外层 stride 和 Storage 偏移；
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

    # 只尝试扩宽；布局是否兼容直接由 view 判断，任一项失败就整批回退。
    for transfer_bytes, transfer_dtype in copy_dtypes:
        if transfer_bytes <= element_bytes:
            break
        try:
            return [(src.view(transfer_dtype), dst.view(transfer_dtype)) for src, dst in copies]
        except RuntimeError:
            continue
    return copies


# ========================================================================
# 四、内部依赖约束
# ========================================================================


@lru_cache(maxsize=1)
def _check_collapse_view_version(torch_version: str) -> None:
    """仅允许已验证的 PyTorch 版本使用内部维度合并接口，正常进程只校验一次。"""
    # _collapse_view_helper 不是稳定公共 API：历史版本的 end 开闭区间及
    # must_be_valid 参数不同，单元素轴的处理也曾变化。升级后须先核对签名、
    # 区间语义并通过 padding/环形边界回归，再修改此版本范围。
    assert torch_version.startswith("2.11."), (
        f"tail dimension merging supports PyTorch 2.11.x, got {torch_version}; "
        "revalidate _collapse_view_helper before updating the supported version"
    )

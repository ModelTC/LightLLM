"""窗口压缩器候选/gate ring 与时间有序快照的 GPU 搬运。"""

import torch
import triton
import triton.language as tl

from ..utils import view_max_copy_dtype


def copy_compressor_window(
    local: torch.Tensor,
    external: torch.Tensor,
    state_token_len: int,
    write: bool,
) -> None:
    """在候选/gate ring 与时间有序快照之间复制，窗口长度由 external 的形状决定。

    大小页可以只保留跨步重叠历史，PD 保存完整窗口；两者均不包含 MTP 候选区。
    """
    if not local.numel():
        return
    local_view, external_view = view_max_copy_dtype(copies=[(local, external)])[0]
    window = external.shape[1]
    elements = local_view.shape[0] * window * local_view.shape[2]
    _copy_compressor_window[(min(24, triton.cdiv(elements, 1024)),)](
        LOCAL=local_view,
        EXTERNAL=external_view,
        window_start=state_token_len - window,
        external_window_start=0,
        LOCAL_STRIDES=local_view.stride(),
        EXTERNAL_STRIDES=external_view.stride(),
        LAYERS=local_view.shape[0],
        LOCAL_RING_SIZE=local.shape[1],
        EXTERNAL_RING_SIZE=window,
        WINDOW_SIZE=window,
        CHANNELS=local_view.shape[2],
        ELEMENT_BYTES=local_view.element_size(),
        LOCAL_CIRCULAR=True,
        EXTERNAL_CIRCULAR=False,
        WRITE=write,
        BLOCK_ELEMENTS=1024,
        num_warps=4,
    )


@triton.jit(do_not_specialize=["window_start", "external_window_start"])
def _copy_compressor_window(
    LOCAL,  # 当前请求的本地 ring：[layer, ring_slot, channel]。
    EXTERNAL,  # 已选定页中的候选/gate 拼接字段：[layer, window, channel]。
    window_start,  # 未取模的绝对窗口起点；短前缀时可为负数，保持为运行时参数。
    external_window_start,  # 本次窗口在外部字段中的起点，通常为 0。
    LOCAL_STRIDES: tl.constexpr,
    EXTERNAL_STRIDES: tl.constexpr,
    LAYERS: tl.constexpr,
    LOCAL_RING_SIZE: tl.constexpr,
    EXTERNAL_RING_SIZE: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    CHANNELS: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    LOCAL_CIRCULAR: tl.constexpr,
    EXTERNAL_CIRCULAR: tl.constexpr,
    WRITE: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
):
    if ELEMENT_BYTES == 8:
        copy_dtype = tl.uint64
    elif ELEMENT_BYTES == 4:
        copy_dtype = tl.uint32
    elif ELEMENT_BYTES == 2:
        copy_dtype = tl.uint16
    else:
        copy_dtype = tl.uint8
    LOCAL = LOCAL.to(tl.pointer_type(copy_dtype))
    EXTERNAL = EXTERNAL.to(tl.pointer_type(copy_dtype))
    window_start = tl.cast(window_start, tl.int64)
    external_window_start = tl.cast(external_window_start, tl.int64)
    element_count = LAYERS * WINDOW_SIZE * CHANNELS
    program_start = tl.program_id(0).to(tl.int64) * BLOCK_ELEMENTS
    grid_stride = tl.num_programs(0).to(tl.int64) * BLOCK_ELEMENTS
    lanes = tl.arange(0, BLOCK_ELEMENTS).to(tl.int64)

    for block_start in tl.range(program_start, element_count, grid_stride, num_stages=1):
        # 两个方向共用逻辑窗口坐标，只在 load/store 时交换数据流向。
        element = block_start + lanes
        valid = element < element_count
        channel = element % CHANNELS
        window_index = element // CHANNELS % WINDOW_SIZE
        layer = element // (CHANNELS * WINDOW_SIZE)
        # 先用绝对位置判断历史是否存在，再映射到物理 ring，避免取模丢失负数信息。
        position = window_start + window_index
        has_history = position >= 0
        local_slot = position
        if LOCAL_CIRCULAR:
            # 窗口不超过 ring 容量；先加容量，使短前缀读回补零也落在非负槽位。
            local_slot = (position + LOCAL_RING_SIZE) % LOCAL_RING_SIZE
        external_slot = external_window_start + window_index
        if EXTERNAL_CIRCULAR:
            external_slot %= EXTERNAL_RING_SIZE

        # 所有 stride 在参与地址乘加前转为 int64，支持请求切片产生的大 layer stride。
        local_offset = (
            layer * tl.cast(LOCAL_STRIDES[0], tl.int64)
            + local_slot * tl.cast(LOCAL_STRIDES[1], tl.int64)
            + channel * tl.cast(LOCAL_STRIDES[2], tl.int64)
        )
        external_offset = (
            layer * tl.cast(EXTERNAL_STRIDES[0], tl.int64)
            + external_slot * tl.cast(EXTERNAL_STRIDES[1], tl.int64)
            + channel * tl.cast(EXTERNAL_STRIDES[2], tl.int64)
        )
        if WRITE:
            value = tl.load(LOCAL + local_offset, mask=valid & has_history, other=0)
            tl.store(EXTERNAL + external_offset, value, mask=valid)
        else:
            value = tl.load(EXTERNAL + external_offset, mask=valid & has_history, other=0)
            tl.store(LOCAL + local_offset, value, mask=valid)

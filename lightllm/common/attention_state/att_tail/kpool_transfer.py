"""K-pool 的 PD 逻辑窗口传输：一次 kernel 完成环形映射及窗口数据搬运。"""

import torch
import triton
import triton.language as tl

from ..types import AxisMapping, StateTransferLayout


# ========================================================================
# 一、K-pool 传输：固定布局 + 本次窗口位置，直接操作原始存储
# ========================================================================


def transfer_kpool_pages(
    tail: torch.Tensor,
    page_buffers: torch.Tensor,
    page_index: int,
    state_token_len: int,
    layout: StateTransferLayout,
    write: bool,
    axis_mapping_overrides: tuple[AxisMapping, ...] = (),
    grid_num: int = 24,
) -> None:
    """在当前 CUDA stream 交换一个请求的 tail 和选定 PD 页，不创建中间 Tensor。

    本地 tail 为 [layer, ring_slot, channel]，ring 容量为 pool_size + 本地 mtp_step；
    外部页保存最近 pool_size 个逻辑位置，按时间顺序排列。P/D 的 ring 容量可以不同。
    窗口起点由 state_token_len 推导；axis_mapping_overrides 可覆盖本次 window 映射，
    其余轴沿用缓存布局。K-pool 只支持 window 环形寻址，channel 保持完整、同序对应。
    两个方向都只遍历逻辑窗口，短前缀对应的窗口内位置补零；
    本地窗口之外的 MTP 槽位保持原值，不在传输时清理。
    调用方负责传输完成依赖及存储生命周期；返回不表示数据已经就绪。
    """
    # 阶段一：解析选定页的 tail 区域，统一为 [layer, window, channel] 视图。
    # K-pool 布局由 State 创建；此处直接使用约定的单个 tail region。
    region = layout.regions[0]
    if write and not region.write_enabled:
        return

    page_bytes = page_buffers[page_index].view(torch.uint8).view(-1)
    field_bytes = page_bytes[region.byte_start:region.byte_end]
    external_tail = field_bytes.view(region.dtype).view(region.shape)
    axis_order = tuple(region.axes.index(axis) for axis in ("layer", "window", "channel"))
    external_tail = external_tail.permute(axis_order)
    local_ring_size = tail.shape[1]
    external_ring_size = external_tail.shape[1]
    channel_count = tail.shape[2]

    # 阶段二：确定本次窗口。显式覆盖的起点直接使用；否则从 prefill 长度推导。
    # 只计算整数位置，不修改 layout，也不创建新的 AxisMapping 对象。
    layout_mappings = {item.axis: item for item in region.axis_mappings}
    mapping_overrides = {item.axis: item for item in axis_mapping_overrides}
    window_mapping = mapping_overrides.get("window", layout_mappings["window"])
    channel_mapping = mapping_overrides.get("channel", layout_mappings["channel"])
    window_size = window_mapping.length
    local_window_start = window_mapping.local_start
    external_window_start = window_mapping.external_start

    if window_mapping.local_circular:
        if "window" not in mapping_overrides:
            # 外部 window[i] 对应绝对位置 state_token_len - window_size + i。
            local_window_start += state_token_len - window_size
        local_window_start %= local_ring_size
    if window_mapping.external_circular:
        external_window_start %= external_ring_size
    invalid_prefix = max(window_size - state_token_len, 0)

    # 定制 kernel 完整处理 key/gate；窗口不能重复覆盖或越过线性存储边界。
    assert (
        channel_mapping.local_start == channel_mapping.external_start == 0
        and channel_mapping.length == channel_count
        and not channel_mapping.local_circular and not channel_mapping.external_circular
    ), "K-pool transfers require the complete channel axis"
    assert (
        0 < window_size <= min(local_ring_size, external_ring_size)
        and (window_mapping.local_circular or 0 <= local_window_start <= local_ring_size - window_size)
        and (window_mapping.external_circular or 0 <= external_window_start <= external_ring_size - window_size)
    ), "window exceeds storage capacity"

    # 阶段三：只扩大连续 channel 的搬运单位，layer/window 维度和环形语义保持不变。
    # 例如 BF16 的 [layers, ring_size, 256] 按 INT64 搬运时变为 [layers, ring_size, 64]。
    local_tail = tail
    if local_tail.stride(-1) == external_tail.stride(-1) == 1:
        element_bytes = local_tail.element_size()
        for unit_bytes, unit_dtype in ((8, torch.int64), (4, torch.int32), (2, torch.int16)):
            if unit_bytes <= element_bytes:
                break
            elements_per_unit = unit_bytes // element_bytes
            # 每个 channel 行都必须能完整分组，不能跨 window 拼接搬运单元。
            if channel_count % elements_per_unit != 0:
                continue
            # Tensor 起点和 storage_offset 都须满足宽类型 view 的对齐要求。
            if (
                local_tail.data_ptr() % unit_bytes != 0
                or external_tail.data_ptr() % unit_bytes != 0
                or local_tail.storage_offset() % elements_per_unit != 0
                or external_tail.storage_offset() % elements_per_unit != 0
            ):
                continue
            # layer/window 的步长必须能换算为新类型的整数元素数。
            if (
                any(stride % elements_per_unit != 0 for stride in local_tail.stride()[:-1])
                or any(stride % elements_per_unit != 0 for stride in external_tail.stride()[:-1])
            ):
                continue
            local_tail = local_tail.view(unit_dtype)
            external_tail = external_tail.view(unit_dtype)
            break

    # 阶段四：正反方向都只处理逻辑窗口，窗口外的 MTP 槽位不参与读写。
    # grid_num 是发射上限，大窗口由 kernel 的跳步循环处理。
    element_count = local_tail.shape[0] * window_size * local_tail.shape[2]
    block_elements = 1024
    program_count = min(grid_num, triton.cdiv(element_count, block_elements))
    _transfer_kpool_pages[(program_count,)](
        LOCAL=local_tail,
        EXTERNAL=external_tail,
        local_window_start=local_window_start,
        external_window_start=external_window_start,
        invalid_prefix=invalid_prefix,
        LOCAL_STRIDES=local_tail.stride(),
        EXTERNAL_STRIDES=external_tail.stride(),
        LAYERS=local_tail.shape[0],
        LOCAL_RING_SIZE=local_ring_size,
        EXTERNAL_RING_SIZE=external_ring_size,
        WINDOW_SIZE=window_size,
        CHANNELS=local_tail.shape[2],
        ELEMENT_BYTES=local_tail.element_size(),
        LOCAL_CIRCULAR=window_mapping.local_circular,
        EXTERNAL_CIRCULAR=window_mapping.external_circular,
        WRITE=write,
        BLOCK_ELEMENTS=block_elements,
        num_warps=4,
    )


# ========================================================================
# 二、定制 kernel：计算窗口位置 → 读取有效历史 → 写出数据或零
# ========================================================================


@triton.jit(do_not_specialize=["local_window_start", "external_window_start", "invalid_prefix"])
def _transfer_kpool_pages(
    LOCAL,  # 当前请求的本地 ring：[layer, ring_slot, channel]。
    EXTERNAL,  # 已选定页中的 tail 字段：[layer, window, channel]。
    local_window_start,  # 本次逻辑窗口在本地 ring 的起点，保持为运行时参数。
    external_window_start,  # 本次窗口在外部字段中的起点，通常为 0。
    invalid_prefix,  # 短前缀中不存在的窗口前部位置数量。
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
    local_window_start = tl.cast(local_window_start, tl.int64)
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
        window = element // CHANNELS % WINDOW_SIZE
        layer = element // (CHANNELS * WINDOW_SIZE)
        local_slot = local_window_start + window
        if LOCAL_CIRCULAR:
            local_slot %= LOCAL_RING_SIZE
        external_slot = external_window_start + window
        if EXTERNAL_CIRCULAR:
            external_slot %= EXTERNAL_RING_SIZE
        has_history = (window >= invalid_prefix) & (window < WINDOW_SIZE)

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

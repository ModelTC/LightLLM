"""MLA 的统一 CPU/PD 页传输：按 token 和页索引直接交换 latent 与外部页。"""

import torch
import triton
import triton.language as tl

from ..types import StateTransferLayout


# ========================================================================
# 一、CPU/PD 共用的页传输入口
# ========================================================================


def transfer_mla_pages(
    kv_storage: torch.Tensor,
    token_indices: torch.Tensor,
    page_buffers: torch.Tensor,
    page_indices: list[int],
    *,
    layout: StateTransferLayout,
    write: bool,
    grid_num: int = 36,
    page_readies: list[bool] | None = None,
) -> None:
    """直接搬运 latent；CPU 传入页列表，PD 传入 [pd_page_index]，同步由调用方负责。

    外部页为 pinned/注册 CPU 内存或当前设备的 CUDA PD buffer。
    单页和多页统一通过页索引 Tensor 定位，共用布局和拷贝 kernel。
    grid_num 控制每个字段发射的 program 数，默认 36；kernel 跳步循环覆盖全部元素。
    """
    assert grid_num > 0, "grid_num must be positive"
    assert kv_storage.is_cuda, "page transfers require CUDA State storage"
    assert page_buffers.ndim >= 2, "page buffers must include a leading page dimension"
    assert isinstance(page_indices, list), "page indices must be a list"
    if page_buffers.device.type == "cpu":
        assert page_buffers.is_pinned(), "CPU pages must be pinned or registered"
    else:
        assert page_buffers.device == kv_storage.device, "CUDA pages must reside on the State device"
    page_tokens = layout.tokens_per_page
    assert page_tokens is not None and page_tokens > 0, "KV transfers require a positive token capacity"
    assert token_indices.is_cuda, "token indices must reside on CUDA"
    token_count = token_indices.numel()
    assert len(page_indices) == triton.cdiv(x=token_count, y=page_tokens), "KV token and page counts differ"
    if write and page_readies is not None:
        assert len(page_readies) == len(page_indices), "page and ready counts differ"
        if all(page_readies):
            return
    if token_count == 0:
        return

    # 阶段一：准备索引，不预选或复制 latent。
    # token_indices[i] 是第 i 个 token 的 GPU 物理槽位，直接使用传入的 dtype 和 stride。
    # 跨 GPU 的 KV 由传输进程通过现有 IPC peer 映射导入当前设备，kernel 沿当前流访问。
    # page_ids[j] 是第 j 个逻辑页对应的外部物理页 ID，允许不连续、不递增。
    # CPU/PD、单页/多页统一通过 pinned 索引异步提交到 GPU，临时显存只与页数有关。
    page_ids = torch.tensor(data=page_indices, dtype=torch.int64, pin_memory=True).to(
        device=kv_storage.device,
        non_blocking=True,
    )

    # ready 标记按逻辑页排列，保留完整 token/page 索引，避免跳过中间页后错位。
    # 本 State 的全部字段共享此 Tensor；PD 和读回路径不需要 ready 标记。
    page_readies_tensor = None
    if write and page_readies is not None:
        page_readies_tensor = torch.tensor(data=page_readies, dtype=torch.bool, pin_memory=True).to(
            device=kv_storage.device,
            non_blocking=True,
        )

    # 阶段二：构造全部字段的视图，完成后再统一提交 kernel。
    # page_bytes 是 [物理页数, 每页可见字节数]，第一维仍保留真实页跨度。
    # 一个联合页可能还保存其他 State 的字段，不能用当前 latent 大小代替页跨度。
    page_bytes = page_buffers.view(torch.uint8).view(page_buffers.shape[0], -1)
    copy_views = []
    for region in layout.regions:
        if write and not region.write_enabled:
            continue
        if region.field == "kv":
            local = kv_storage
            axes = ("layer", "token", "head", "dim")
        else:
            raise AssertionError(f"unknown transfer field {region.field}")
        # 从各页中取字段的字节区间，按 dtype 和布局形状解释，不搬运数据。
        external = page_bytes[:, region.byte_start:region.byte_end].view(region.dtype)
        external = external.view(page_buffers.shape[0], *region.shape)
        # 外部统一为 [page, layer, page_token, head, field_width]；+1 是因为多了 page 轴。
        external = external.permute(0, *(region.axes.index(axis) + 1 for axis in axes))
        local_slices = [slice(None)] * 4
        external_slices = [slice(None)] * 5
        for axis_slice in region.axis_slices:
            # src_start/dst_start 固定表示本地/外部起点，不随读写方向改变。
            # 例：本地 heads [0, 2) 对应外部 heads [4, 6)，切片保留两侧真实 stride。
            axis = axes.index(axis_slice.axis)
            local_slices[axis] = slice(axis_slice.src_start, axis_slice.src_start + axis_slice.length)
            external_slices[axis + 1] = slice(axis_slice.dst_start, axis_slice.dst_start + axis_slice.length)
        local = local[tuple(local_slices)]
        external = external[tuple(external_slices)]
        # token 轴不比较：local 是 GPU 槽位容量，external 是单页容量。
        assert local.dtype == external.dtype
        assert (local.shape[0], local.shape[2], local.shape[3]) == (
            external.shape[1], external.shape[3], external.shape[4]
        ), "local and external copy views must have the same shape"
        copy_views.append((local, external))

    # 阶段三：每个 传输字段一次 launch，覆盖全部层、有效 token 和外部页。
    # elements 是标量元素数，不是字节数；每个字段固定发射 grid_num 个 program。
    # 超出一轮 grid 覆盖范围的数据由 kernel 跳步处理，没有 gather/scatter 中间存储。
    # stream/event 同步由调用方负责，函数返回不表示 CPU 页已经可读取或复用。
    for local, external in copy_views:
        elements = local.shape[0] * token_count * local.shape[2] * local.shape[3]
        _copy_mla_pages[(grid_num,)](
            LOCAL=local,
            EXTERNAL=external,
            TOKEN_IDS=token_indices,
            TOKEN_STRIDE=token_indices.stride(0),
            PAGE_IDS=page_ids,
            PAGE_READIES=page_readies_tensor,
            HAS_PAGE_READIES=page_readies_tensor is not None,
            token_count=token_count,
            element_count=elements,
            LOCAL_STRIDES=local.stride(),
            EXTERNAL_STRIDES=external.stride(),
            PAGE_TOKENS=page_tokens,
            HEADS=local.shape[2],
            WIDTH=local.shape[3],
            ELEMENT_BYTES=local.element_size(),
            WRITE=write,
            BLOCK=256,
        )


# ========================================================================
# 二、CUDA kernel：按索引和实际 stride 双向搬运原始编码
# ========================================================================


@triton.jit
def _copy_mla_pages(
    # 数据指针：一次处理一个字段；调用方已应用 TP 切片，两侧 dtype 相同。
    LOCAL,  # GPU 视图：[layer, token_slot, head, dim]。
    EXTERNAL,  # CPU/PD 外部页视图：[page, layer, page_token, head, field_width]。
    # 索引指针：均位于 GPU；索引值是物理槽位/页 ID，允许不连续、不递增。
    TOKEN_IDS,  # TOKEN_IDS[token] 指向 GPU KV 存储中的 token 槽位。
    PAGE_IDS,  # GPU 页索引 Tensor，PAGE_IDS[token // PAGE_TOKENS] 给出外部物理页 ID。
    PAGE_READIES,  # 可选的 GPU bool 数组；True 表示该逻辑页已就绪，禁止覆盖。
    # 本次搬运范围：只覆盖有效 token，不包含尾页空余位置。
    token_count,  # 本次有效 token 数；调用方保证大于 0。
    element_count,  # layer 数 × token_count × HEADS × WIDTH，单位为标量元素。
    # 编译期寻址配置：stride 的单位是元素数，不是字节数。
    TOKEN_STRIDE: tl.constexpr,  # token 索引步长，直接支持切片视图，不创建连续副本。
    LOCAL_STRIDES: tl.constexpr,  # 对应 LOCAL 的四个轴，保留原 GPU 存储步长。
    EXTERNAL_STRIDES: tl.constexpr,  # 对应 EXTERNAL 的五个轴，包含真实外部页跨度。
    PAGE_TOKENS: tl.constexpr,  # 每个外部页可容纳的 token 数。
    HEADS: tl.constexpr,  # 本次 TP 切片参与拷贝的 head 数。
    WIDTH: tl.constexpr,  # 字段末轴参与拷贝的元素数，如 dim、group 或 scale。
    # 编译期搬运配置。
    ELEMENT_BYTES: tl.constexpr,  # 单个元素字节数，用于选择同宽整数指针。
    HAS_PAGE_READIES: tl.constexpr,  # 无 ready 标记时编译掉检查，保持 PD/读回路径。
    WRITE: tl.constexpr,  # True：KV→外部页；False：外部页→KV。
    BLOCK: tl.constexpr,  # 每个 program 处理的标量元素数，当前调用方传入 256。
):
    # 阶段一：把两侧指针解释为同宽整数指针，逐位保留数据（包括 NaN 位模式）。
    # 这里只改变指针类型，不进行数值转换；元素宽度不变，因此原 stride 仍然有效。
    if ELEMENT_BYTES == 1:
        LOCAL = LOCAL.to(dtype=tl.pointer_type(element_ty=tl.uint8))
        EXTERNAL = EXTERNAL.to(dtype=tl.pointer_type(element_ty=tl.uint8))
    elif ELEMENT_BYTES == 2:
        LOCAL = LOCAL.to(dtype=tl.pointer_type(element_ty=tl.uint16))
        EXTERNAL = EXTERNAL.to(dtype=tl.pointer_type(element_ty=tl.uint16))
    elif ELEMENT_BYTES == 4:
        LOCAL = LOCAL.to(dtype=tl.pointer_type(element_ty=tl.uint32))
        EXTERNAL = EXTERNAL.to(dtype=tl.pointer_type(element_ty=tl.uint32))
    else:
        LOCAL = LOCAL.to(dtype=tl.pointer_type(element_ty=tl.uint64))
        EXTERNAL = EXTERNAL.to(dtype=tl.pointer_type(element_ty=tl.uint64))

    # 阶段二：确定当前 program 的起点和整张 grid 的步长。
    # 每轮处理 BLOCK 个元素，下一轮前进 grid_num * BLOCK，所有 program 共同覆盖完整范围。
    # 使用 int64 避免大容量寻址溢出；tl.cast 兼容 token_count=1 被特化为常量的情况。
    token_count = tl.cast(input=token_count, dtype=tl.int64)
    program_start = tl.program_id(axis=0).to(dtype=tl.int64) * BLOCK
    grid_stride = tl.num_programs(axis=0).to(dtype=tl.int64) * BLOCK
    block_offsets = tl.arange(start=0, end=BLOCK)
    # 起点超出 element_count 的 program 不进入循环；最后一块通过 valid 屏蔽尾部元素。
    for block_start in range(program_start, element_count, grid_stride):
        offsets = block_start + block_offsets
        valid = offsets < element_count
        token = offsets // (WIDTH * HEADS) % token_count

        # 阶段三：把逻辑 token 位置映射到两侧物理存储。
        # slot 是 GPU token 槽位；page 是外部物理页；token % PAGE_TOKENS 是页内位置。
        # 例如 PAGE_TOKENS=16 时，逻辑 token 18 使用 TOKEN_IDS[18] 和 PAGE_IDS[1] 的第 2 个位置。
        # 索引读取同样使用 valid，避免尾部无效元素访问 TOKEN_IDS/PAGE_IDS 越界。
        # 写出 ready 页时同时屏蔽源 KV 读取和目标写入，不仅仅跳过 store。
        copy_block = True
        if WRITE and HAS_PAGE_READIES:
            ready = tl.load(pointer=PAGE_READIES + token // PAGE_TOKENS, mask=valid, other=1)
            valid = valid & ~ready
            # 一个 block 可能跨页：混合状态保留逐元素 mask；全部无需写出则跳过后续操作。
            copy_block = tl.sum(input=valid.to(dtype=tl.int32), axis=0) > 0
        if copy_block:
            dim = offsets % WIDTH
            head = offsets // WIDTH % HEADS
            layer = offsets // (WIDTH * HEADS * token_count)
            slot = tl.load(pointer=TOKEN_IDS + token * TOKEN_STRIDE, mask=valid, other=0).to(dtype=tl.int64)
            page = tl.load(pointer=PAGE_IDS + token // PAGE_TOKENS, mask=valid, other=0).to(dtype=tl.int64)

            # 阶段四：按各轴真实 stride 计算元素偏移，兼容 TP 切片及轴顺序调整后的视图。
            # EXTERNAL 已指向当前字段在联合页中的起点；这里无需再次加 region.byte_start。
            # EXTERNAL_STRIDES[0] 使用整个外部页的跨度，不能替换为当前 传输字段的大小。
            local_offset = (
                layer * LOCAL_STRIDES[0]
                + slot * LOCAL_STRIDES[1]
                + head * LOCAL_STRIDES[2]
                + dim * LOCAL_STRIDES[3]
            )
            external_offset = (
                page * EXTERNAL_STRIDES[0]
                + layer * EXTERNAL_STRIDES[1]
                + (token % PAGE_TOKENS) * EXTERNAL_STRIDES[2]
                + head * EXTERNAL_STRIDES[3]
                + dim * EXTERNAL_STRIDES[4]
            )

            # 阶段五：按方向直接读写两侧存储，不生成 gather/scatter 中间 Tensor。
            # 两个方向使用相同的索引和有效范围；尾页空余、其他 TP 分片及其他字段保持不变。
            if WRITE:
                value = tl.load(pointer=LOCAL + local_offset, mask=valid, other=0)
                tl.store(pointer=EXTERNAL + external_offset, value=value, mask=valid)
            else:
                value = tl.load(pointer=EXTERNAL + external_offset, mask=valid, other=0)
                tl.store(pointer=LOCAL + local_offset, value=value, mask=valid)

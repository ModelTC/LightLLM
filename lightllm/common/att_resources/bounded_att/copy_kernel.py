"""将 bounded attention 的同一形状的多页视图聚合到 CUDA stream 拷贝。"""

import torch
import triton
import triton.language as tl


# ========================================================================
# 一、聚合入口：只上传指针表，shape/stride 作为常量
# ========================================================================


def copy_bounded_att_tensors(
    copies: list[tuple[torch.Tensor, torch.Tensor]],
    grid_num: int = 24,
    num_warps: int = 8,
    num_stages: int = 1,
) -> None:
    """在当前 CUDA stream 上聚合复制同一形状的视图，默认 24 个 program；warp 数与循环流水深度可按搬运场景覆盖。

    num_stages 控制 tl.range 的循环流水深度，默认 1；提高该值不保证所有方向的带宽都增加。
    支持 CUDA/IPC 存储和本进程已 pinned/registered 的 CPU 存储，两端均为
    CPU 页池时也走 GPU。整批所有源、目标的形状和 dtype 必须相同，支持最多六维的 stride。
    同一侧的全部视图共用 stride，源与目标之间可不同；入口只核对首尾项。
    只合并两端共同连续的尾部维度，并按对齐条件创建最大 8 字节的整数视图。
    这些 view 不复制数据；不满足合并或宽类型条件时保留原布局，不转换数值。
    每项表示独立复制，目标区域不得相互覆盖，也不得覆盖本批仍需读取的源数据。
    直接使用 data_ptr()；调用方保证地址可被当前 GPU 访问，并持有存储直到拷贝完成。
    源数据就绪，以及 CPU 读取或复用页面前的完成事件，也由调用方管理。
    """
    if not copies:
        return
    assert grid_num > 0, "copy grid must be positive"
    # 阶段一：只合并两端共同连续的尾部维度，保留前面的布局。
    copies = _view_min_dimensions(copies=copies)
    # 阶段二：按整批地址和行布局选择最宽的搬运单元，不转换数值。
    copies = _view_max_dtype(copies=copies)
    # 阶段三：上传两张指针表，按最终视图维数发射聚合拷贝算子。
    _launch_copy_kernel(
        copies=copies,
        grid_num=grid_num,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# ========================================================================
# 二、视图整理与算子发射：降维 → 扩大搬运单元 → kernel
# ========================================================================


def _view_min_dimensions(
    copies: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """按首对布局合并连续尾维，遇到任一侧不连续就停止。"""
    # 每批复制使用同一种 shape；由调用方分别收集 Conv、SSM 或同一传输区域。
    source, destination = copies[0]
    last_source, last_destination = copies[-1]
    # 整批布局由调用方保证一致，只检查首尾两对 Tensor。
    assert source.shape == destination.shape == last_source.shape == last_destination.shape, "copy shapes must match"
    assert source.dtype == destination.dtype == last_source.dtype == last_destination.dtype, "copy dtypes must match"
    ndim = source.ndim
    assert 1 <= ndim <= 6, "copy tensors must have one to six dimensions"
    assert source.numel() > 0, "copy tensors must be nonempty"
    shape = tuple(source.shape)
    # 同一侧所有视图共用 stride；只核对首尾，源和目标之间允许不同布局。
    src_stride = source.stride()
    dst_stride = destination.stride()
    assert src_stride == last_source.stride(), "source strides must match across the batch"
    assert dst_stride == last_destination.stride(), "destination strides must match across the batch"
    # 从最后一维向前扩展连续区间；任一侧有间隙就停止，不再处理前面的轴。
    # 例如 (2, 3, 4, 5) 的末轴带 padding 时保持原形状，不额外合并前面的 (2, 3, 4)。
    first_tail_axis = ndim - 1
    tail_elements = shape[-1]
    while first_tail_axis > 0:
        outer_axis = first_tail_axis - 1
        if (
            src_stride[outer_axis] != src_stride[first_tail_axis] * shape[first_tail_axis]
            or dst_stride[outer_axis] != dst_stride[first_tail_axis] * shape[first_tail_axis]
        ):
            break
        tail_elements *= shape[outer_axis]
        first_tail_axis = outer_axis
    if first_tail_axis == ndim - 1:
        return copies
    transfer_shape = shape[:first_tail_axis] + (tail_elements,)
    return [(source.view(transfer_shape), destination.view(transfer_shape)) for source, destination in copies]


def _view_max_dtype(
    copies: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """将整批 view 为最大可用整数宽度，保留原始位模式和存储间隙。"""
    source, destination = copies[0]
    if source.stride(-1) != 1 or destination.stride(-1) != 1:
        return copies

    # 每个切片的起点可能不同；地址和 storage_offset 都须支持相同的宽类型 view。
    # 累计低位即可检查整批对齐，不能仅检查首尾，否则会漏掉中间错位的视图。
    pointer_alignment_bits = 0
    offset_alignment_bits = 0
    for src_tensor, dst_tensor in copies:
        pointer_alignment_bits |= src_tensor.data_ptr() | dst_tensor.data_ptr()
        offset_alignment_bits |= src_tensor.storage_offset() | dst_tensor.storage_offset()

    # 最内维连续时，将相邻元素作为一个更宽的整数搬运；优先 8 字节，其次 4/2 字节。
    # 行宽、外层 stride、storage_offset 及整批地址都须对齐，避免跨行 padding 或错位读取。
    element_bytes = source.element_size()
    for transfer_bytes, transfer_dtype in ((8, torch.int64), (4, torch.int32), (2, torch.int16)):
        if transfer_bytes <= element_bytes:
            break
        elements_per_unit = transfer_bytes // element_bytes
        if (
            pointer_alignment_bits % transfer_bytes != 0
            or source.shape[-1] % elements_per_unit != 0
            or offset_alignment_bits % elements_per_unit != 0
            or any(stride % elements_per_unit for stride in source.stride()[:-1])
            or any(stride % elements_per_unit for stride in destination.stride()[:-1])
        ):
            continue
        return [(src.view(transfer_dtype), dst.view(transfer_dtype)) for src, dst in copies]
    return copies


def _launch_copy_kernel(
    copies: list[tuple[torch.Tensor, torch.Tensor]],
    grid_num: int,
    num_warps: int,
    num_stages: int,
) -> None:
    """使用整理后的视图地址和共享布局，在当前流上发射一次聚合拷贝。"""
    # 确保当前设备上下文已经建立；纯 CPU 页池之间的复制也使用这个设备和流。
    stream = torch.cuda.current_stream()
    source, destination = copies[0]
    source_pointers, destination_pointers = [], []
    for src_tensor, dst_tensor in copies:
        source_pointers.append(src_tensor.data_ptr())
        destination_pointers.append(dst_tensor.data_ptr())

    # 整批共享 shape/dtype 和同侧 stride，首对最终视图提供全部编译期布局参数。
    ndim = source.ndim
    shape = tuple(source.shape)
    src_stride = source.stride()
    dst_stride = destination.stride()
    unsigned_dtypes = {1: tl.uint8, 2: tl.uint16, 4: tl.uint32, 8: tl.uint64}

    kernels = {
        1: _copy_bounded_att_1d,
        2: _copy_bounded_att_2d,
        3: _copy_bounded_att_3d,
        4: _copy_bounded_att_4d,
        5: _copy_bounded_att_5d,
        6: _copy_bounded_att_6d,
    }
    # 源、目标地址分别上传为独立 Tensor，同一位置对应一对复制视图。
    source_pointer_tensor = torch.tensor(
        data=source_pointers,
        dtype=torch.int64,
        device="cpu",
        pin_memory=True,
    ).to(device=stream.device, non_blocking=True)
    destination_pointer_tensor = torch.tensor(
        data=destination_pointers,
        dtype=torch.int64,
        device="cpu",
        pin_memory=True,
    ).to(device=stream.device, non_blocking=True)
    # 固定 grid=24 的跨场景调优使用每轮 1024 个搬运单元，兼顾小块延迟和大块吞吐。
    kernels[ndim][(grid_num,)](
        SRC_POINTERS=source_pointer_tensor,
        DST_POINTERS=destination_pointer_tensor,
        total_element_count=source.numel() * len(copies),
        SHAPE=shape,
        SRC_STRIDES=src_stride,
        DST_STRIDES=dst_stride,
        DTYPE=unsigned_dtypes[source.element_size()],
        BLOCK_ELEMENTS=1024,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
    )


# ========================================================================
# 三、分维数聚合算子：计算索引与偏移 → 读取指针 → load 数据 → store 数据
# ========================================================================


@triton.jit
def _copy_bounded_att_1d(
    SRC_POINTERS,
    DST_POINTERS,
    total_element_count,
    SHAPE: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 阶段一：grid 按整批总元素数跳步，每轮处理 BLOCK_ELEMENTS 个元素。
    # 先转 int64 再计算起点和步长，避免大批量传输时索引溢出。
    total_element_count = tl.cast(input=total_element_count, dtype=tl.int64)
    program_start = tl.program_id(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    grid_stride = tl.num_programs(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    element_lanes = tl.arange(start=0, end=BLOCK_ELEMENTS)
    elements_per_copy = SHAPE[0]
    # stride 在参与乘加前显式转为 int64，避免大跨度地址计算发生 32 位溢出。
    src_stride_0 = tl.cast(input=SRC_STRIDES[0], dtype=tl.int64)
    dst_stride_0 = tl.cast(input=DST_STRIDES[0], dtype=tl.int64)
    # 循环级流水深度：允许编译器重叠相邻迭代的访存，与 launch 的 num_stages 选项不同。
    for block_start in tl.range(program_start, total_element_count, grid_stride, num_stages=NUM_STAGES):
        # 阶段二：全局索引拆成复制项编号和项内元素索引，再还原各维坐标。
        # 一个 block 可以跨越多对 Tensor；尾部 mask 同时保护指针表和数据访问。
        global_indices = block_start + element_lanes
        valid_elements = global_indices < total_element_count
        copy_index = global_indices // elements_per_copy
        element_indices = global_indices % elements_per_copy
        # 一维直接按元素序号应用两端 stride。
        axis0 = element_indices
        source_offsets = axis0 * src_stride_0
        destination_offsets = axis0 * dst_stride_0

        # 阶段三：按元素所属复制项，从两张指针表读取源/目标地址，按元素宽度解释指针。
        # 使用等宽无符号类型保持原始位模式，不进行浮点数值转换。
        source_address = tl.load(pointer=SRC_POINTERS + copy_index, mask=valid_elements, other=0)
        destination_address = tl.load(pointer=DST_POINTERS + copy_index, mask=valid_elements, other=0)
        source_ptr = source_address.to(dtype=tl.pointer_type(element_ty=DTYPE))
        destination_ptr = destination_address.to(dtype=tl.pointer_type(element_ty=DTYPE))

        # 阶段四：读取源数据并写入目标，末尾越界元素不参与 load/store。
        values = tl.load(pointer=source_ptr + source_offsets, mask=valid_elements, other=0)
        tl.store(pointer=destination_ptr + destination_offsets, value=values, mask=valid_elements)


@triton.jit
def _copy_bounded_att_2d(
    SRC_POINTERS,
    DST_POINTERS,
    total_element_count,
    SHAPE: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 阶段一：grid 按整批总元素数跳步，每轮处理 BLOCK_ELEMENTS 个元素。
    # 先转 int64 再计算起点和步长，避免大批量传输时索引溢出。
    total_element_count = tl.cast(input=total_element_count, dtype=tl.int64)
    program_start = tl.program_id(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    grid_stride = tl.num_programs(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    element_lanes = tl.arange(start=0, end=BLOCK_ELEMENTS)
    elements_per_copy = SHAPE[0] * SHAPE[1]
    # stride 在参与乘加前显式转为 int64，避免大跨度地址计算发生 32 位溢出。
    src_stride_0 = tl.cast(input=SRC_STRIDES[0], dtype=tl.int64)
    dst_stride_0 = tl.cast(input=DST_STRIDES[0], dtype=tl.int64)
    src_stride_1 = tl.cast(input=SRC_STRIDES[1], dtype=tl.int64)
    dst_stride_1 = tl.cast(input=DST_STRIDES[1], dtype=tl.int64)
    # 循环级流水深度：允许编译器重叠相邻迭代的访存，与 launch 的 num_stages 选项不同。
    for block_start in tl.range(program_start, total_element_count, grid_stride, num_stages=NUM_STAGES):
        # 阶段二：全局索引拆成复制项编号和项内元素索引，再还原各维坐标。
        # 一个 block 可以跨越多对 Tensor；尾部 mask 同时保护指针表和数据访问。
        global_indices = block_start + element_lanes
        valid_elements = global_indices < total_element_count
        copy_index = global_indices // elements_per_copy
        element_indices = global_indices % elements_per_copy
        # 从最内层轴开始还原 2 维坐标；两端可以具有不同的物理 stride。
        axis1 = element_indices % SHAPE[1]
        remaining_indices = element_indices // SHAPE[1]
        axis0 = remaining_indices
        source_offsets = axis0 * src_stride_0 + axis1 * src_stride_1
        destination_offsets = axis0 * dst_stride_0 + axis1 * dst_stride_1

        # 阶段三：按元素所属复制项，从两张指针表读取源/目标地址，按元素宽度解释指针。
        # 使用等宽无符号类型保持原始位模式，不进行浮点数值转换。
        source_address = tl.load(pointer=SRC_POINTERS + copy_index, mask=valid_elements, other=0)
        destination_address = tl.load(pointer=DST_POINTERS + copy_index, mask=valid_elements, other=0)
        source_ptr = source_address.to(dtype=tl.pointer_type(element_ty=DTYPE))
        destination_ptr = destination_address.to(dtype=tl.pointer_type(element_ty=DTYPE))

        # 阶段四：读取源数据并写入目标，末尾越界元素不参与 load/store。
        values = tl.load(pointer=source_ptr + source_offsets, mask=valid_elements, other=0)
        tl.store(pointer=destination_ptr + destination_offsets, value=values, mask=valid_elements)


@triton.jit
def _copy_bounded_att_3d(
    SRC_POINTERS,
    DST_POINTERS,
    total_element_count,
    SHAPE: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 阶段一：grid 按整批总元素数跳步，每轮处理 BLOCK_ELEMENTS 个元素。
    # 先转 int64 再计算起点和步长，避免大批量传输时索引溢出。
    total_element_count = tl.cast(input=total_element_count, dtype=tl.int64)
    program_start = tl.program_id(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    grid_stride = tl.num_programs(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    element_lanes = tl.arange(start=0, end=BLOCK_ELEMENTS)
    elements_per_copy = SHAPE[0] * SHAPE[1] * SHAPE[2]
    # stride 在参与乘加前显式转为 int64，避免大跨度地址计算发生 32 位溢出。
    src_stride_0 = tl.cast(input=SRC_STRIDES[0], dtype=tl.int64)
    dst_stride_0 = tl.cast(input=DST_STRIDES[0], dtype=tl.int64)
    src_stride_1 = tl.cast(input=SRC_STRIDES[1], dtype=tl.int64)
    dst_stride_1 = tl.cast(input=DST_STRIDES[1], dtype=tl.int64)
    src_stride_2 = tl.cast(input=SRC_STRIDES[2], dtype=tl.int64)
    dst_stride_2 = tl.cast(input=DST_STRIDES[2], dtype=tl.int64)
    # 循环级流水深度：允许编译器重叠相邻迭代的访存，与 launch 的 num_stages 选项不同。
    for block_start in tl.range(program_start, total_element_count, grid_stride, num_stages=NUM_STAGES):
        # 阶段二：全局索引拆成复制项编号和项内元素索引，再还原各维坐标。
        # 一个 block 可以跨越多对 Tensor；尾部 mask 同时保护指针表和数据访问。
        global_indices = block_start + element_lanes
        valid_elements = global_indices < total_element_count
        copy_index = global_indices // elements_per_copy
        element_indices = global_indices % elements_per_copy
        # 从最内层轴开始还原 3 维坐标；两端可以具有不同的物理 stride。
        axis2 = element_indices % SHAPE[2]
        remaining_indices = element_indices // SHAPE[2]
        axis1 = remaining_indices % SHAPE[1]
        remaining_indices = remaining_indices // SHAPE[1]
        axis0 = remaining_indices
        source_offsets = (
            axis0 * src_stride_0
            + axis1 * src_stride_1
            + axis2 * src_stride_2
        )
        destination_offsets = (
            axis0 * dst_stride_0
            + axis1 * dst_stride_1
            + axis2 * dst_stride_2
        )

        # 阶段三：按元素所属复制项，从两张指针表读取源/目标地址，按元素宽度解释指针。
        # 使用等宽无符号类型保持原始位模式，不进行浮点数值转换。
        source_address = tl.load(pointer=SRC_POINTERS + copy_index, mask=valid_elements, other=0)
        destination_address = tl.load(pointer=DST_POINTERS + copy_index, mask=valid_elements, other=0)
        source_ptr = source_address.to(dtype=tl.pointer_type(element_ty=DTYPE))
        destination_ptr = destination_address.to(dtype=tl.pointer_type(element_ty=DTYPE))

        # 阶段四：读取源数据并写入目标，末尾越界元素不参与 load/store。
        values = tl.load(pointer=source_ptr + source_offsets, mask=valid_elements, other=0)
        tl.store(pointer=destination_ptr + destination_offsets, value=values, mask=valid_elements)


@triton.jit
def _copy_bounded_att_4d(
    SRC_POINTERS,
    DST_POINTERS,
    total_element_count,
    SHAPE: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 阶段一：grid 按整批总元素数跳步，每轮处理 BLOCK_ELEMENTS 个元素。
    # 先转 int64 再计算起点和步长，避免大批量传输时索引溢出。
    total_element_count = tl.cast(input=total_element_count, dtype=tl.int64)
    program_start = tl.program_id(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    grid_stride = tl.num_programs(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    element_lanes = tl.arange(start=0, end=BLOCK_ELEMENTS)
    elements_per_copy = SHAPE[0] * SHAPE[1] * SHAPE[2] * SHAPE[3]
    # stride 在参与乘加前显式转为 int64，避免大跨度地址计算发生 32 位溢出。
    src_stride_0 = tl.cast(input=SRC_STRIDES[0], dtype=tl.int64)
    dst_stride_0 = tl.cast(input=DST_STRIDES[0], dtype=tl.int64)
    src_stride_1 = tl.cast(input=SRC_STRIDES[1], dtype=tl.int64)
    dst_stride_1 = tl.cast(input=DST_STRIDES[1], dtype=tl.int64)
    src_stride_2 = tl.cast(input=SRC_STRIDES[2], dtype=tl.int64)
    dst_stride_2 = tl.cast(input=DST_STRIDES[2], dtype=tl.int64)
    src_stride_3 = tl.cast(input=SRC_STRIDES[3], dtype=tl.int64)
    dst_stride_3 = tl.cast(input=DST_STRIDES[3], dtype=tl.int64)
    # 循环级流水深度：允许编译器重叠相邻迭代的访存，与 launch 的 num_stages 选项不同。
    for block_start in tl.range(program_start, total_element_count, grid_stride, num_stages=NUM_STAGES):
        # 阶段二：全局索引拆成复制项编号和项内元素索引，再还原各维坐标。
        # 一个 block 可以跨越多对 Tensor；尾部 mask 同时保护指针表和数据访问。
        global_indices = block_start + element_lanes
        valid_elements = global_indices < total_element_count
        copy_index = global_indices // elements_per_copy
        element_indices = global_indices % elements_per_copy
        # 从最内层轴开始还原 4 维坐标；两端可以具有不同的物理 stride。
        axis3 = element_indices % SHAPE[3]
        remaining_indices = element_indices // SHAPE[3]
        axis2 = remaining_indices % SHAPE[2]
        remaining_indices = remaining_indices // SHAPE[2]
        axis1 = remaining_indices % SHAPE[1]
        remaining_indices = remaining_indices // SHAPE[1]
        axis0 = remaining_indices
        source_offsets = (
            axis0 * src_stride_0
            + axis1 * src_stride_1
            + axis2 * src_stride_2
            + axis3 * src_stride_3
        )
        destination_offsets = (
            axis0 * dst_stride_0
            + axis1 * dst_stride_1
            + axis2 * dst_stride_2
            + axis3 * dst_stride_3
        )

        # 阶段三：按元素所属复制项，从两张指针表读取源/目标地址，按元素宽度解释指针。
        # 使用等宽无符号类型保持原始位模式，不进行浮点数值转换。
        source_address = tl.load(pointer=SRC_POINTERS + copy_index, mask=valid_elements, other=0)
        destination_address = tl.load(pointer=DST_POINTERS + copy_index, mask=valid_elements, other=0)
        source_ptr = source_address.to(dtype=tl.pointer_type(element_ty=DTYPE))
        destination_ptr = destination_address.to(dtype=tl.pointer_type(element_ty=DTYPE))

        # 阶段四：读取源数据并写入目标，末尾越界元素不参与 load/store。
        values = tl.load(pointer=source_ptr + source_offsets, mask=valid_elements, other=0)
        tl.store(pointer=destination_ptr + destination_offsets, value=values, mask=valid_elements)


@triton.jit
def _copy_bounded_att_5d(
    SRC_POINTERS,
    DST_POINTERS,
    total_element_count,
    SHAPE: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 阶段一：grid 按整批总元素数跳步，每轮处理 BLOCK_ELEMENTS 个元素。
    # 先转 int64 再计算起点和步长，避免大批量传输时索引溢出。
    total_element_count = tl.cast(input=total_element_count, dtype=tl.int64)
    program_start = tl.program_id(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    grid_stride = tl.num_programs(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    element_lanes = tl.arange(start=0, end=BLOCK_ELEMENTS)
    elements_per_copy = SHAPE[0] * SHAPE[1] * SHAPE[2] * SHAPE[3] * SHAPE[4]
    # stride 在参与乘加前显式转为 int64，避免大跨度地址计算发生 32 位溢出。
    src_stride_0 = tl.cast(input=SRC_STRIDES[0], dtype=tl.int64)
    dst_stride_0 = tl.cast(input=DST_STRIDES[0], dtype=tl.int64)
    src_stride_1 = tl.cast(input=SRC_STRIDES[1], dtype=tl.int64)
    dst_stride_1 = tl.cast(input=DST_STRIDES[1], dtype=tl.int64)
    src_stride_2 = tl.cast(input=SRC_STRIDES[2], dtype=tl.int64)
    dst_stride_2 = tl.cast(input=DST_STRIDES[2], dtype=tl.int64)
    src_stride_3 = tl.cast(input=SRC_STRIDES[3], dtype=tl.int64)
    dst_stride_3 = tl.cast(input=DST_STRIDES[3], dtype=tl.int64)
    src_stride_4 = tl.cast(input=SRC_STRIDES[4], dtype=tl.int64)
    dst_stride_4 = tl.cast(input=DST_STRIDES[4], dtype=tl.int64)
    # 循环级流水深度：允许编译器重叠相邻迭代的访存，与 launch 的 num_stages 选项不同。
    for block_start in tl.range(program_start, total_element_count, grid_stride, num_stages=NUM_STAGES):
        # 阶段二：全局索引拆成复制项编号和项内元素索引，再还原各维坐标。
        # 一个 block 可以跨越多对 Tensor；尾部 mask 同时保护指针表和数据访问。
        global_indices = block_start + element_lanes
        valid_elements = global_indices < total_element_count
        copy_index = global_indices // elements_per_copy
        element_indices = global_indices % elements_per_copy
        # 从最内层轴开始还原 5 维坐标；两端可以具有不同的物理 stride。
        axis4 = element_indices % SHAPE[4]
        remaining_indices = element_indices // SHAPE[4]
        axis3 = remaining_indices % SHAPE[3]
        remaining_indices = remaining_indices // SHAPE[3]
        axis2 = remaining_indices % SHAPE[2]
        remaining_indices = remaining_indices // SHAPE[2]
        axis1 = remaining_indices % SHAPE[1]
        remaining_indices = remaining_indices // SHAPE[1]
        axis0 = remaining_indices
        source_offsets = (
            axis0 * src_stride_0
            + axis1 * src_stride_1
            + axis2 * src_stride_2
            + axis3 * src_stride_3
            + axis4 * src_stride_4
        )
        destination_offsets = (
            axis0 * dst_stride_0
            + axis1 * dst_stride_1
            + axis2 * dst_stride_2
            + axis3 * dst_stride_3
            + axis4 * dst_stride_4
        )

        # 阶段三：按元素所属复制项，从两张指针表读取源/目标地址，按元素宽度解释指针。
        # 使用等宽无符号类型保持原始位模式，不进行浮点数值转换。
        source_address = tl.load(pointer=SRC_POINTERS + copy_index, mask=valid_elements, other=0)
        destination_address = tl.load(pointer=DST_POINTERS + copy_index, mask=valid_elements, other=0)
        source_ptr = source_address.to(dtype=tl.pointer_type(element_ty=DTYPE))
        destination_ptr = destination_address.to(dtype=tl.pointer_type(element_ty=DTYPE))

        # 阶段四：读取源数据并写入目标，末尾越界元素不参与 load/store。
        values = tl.load(pointer=source_ptr + source_offsets, mask=valid_elements, other=0)
        tl.store(pointer=destination_ptr + destination_offsets, value=values, mask=valid_elements)


@triton.jit
def _copy_bounded_att_6d(
    SRC_POINTERS,
    DST_POINTERS,
    total_element_count,
    SHAPE: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 阶段一：grid 按整批总元素数跳步，每轮处理 BLOCK_ELEMENTS 个元素。
    # 先转 int64 再计算起点和步长，避免大批量传输时索引溢出。
    total_element_count = tl.cast(input=total_element_count, dtype=tl.int64)
    program_start = tl.program_id(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    grid_stride = tl.num_programs(axis=0).to(dtype=tl.int64) * BLOCK_ELEMENTS
    element_lanes = tl.arange(start=0, end=BLOCK_ELEMENTS)
    elements_per_copy = SHAPE[0] * SHAPE[1] * SHAPE[2] * SHAPE[3] * SHAPE[4] * SHAPE[5]
    # stride 在参与乘加前显式转为 int64，避免大跨度地址计算发生 32 位溢出。
    src_stride_0 = tl.cast(input=SRC_STRIDES[0], dtype=tl.int64)
    dst_stride_0 = tl.cast(input=DST_STRIDES[0], dtype=tl.int64)
    src_stride_1 = tl.cast(input=SRC_STRIDES[1], dtype=tl.int64)
    dst_stride_1 = tl.cast(input=DST_STRIDES[1], dtype=tl.int64)
    src_stride_2 = tl.cast(input=SRC_STRIDES[2], dtype=tl.int64)
    dst_stride_2 = tl.cast(input=DST_STRIDES[2], dtype=tl.int64)
    src_stride_3 = tl.cast(input=SRC_STRIDES[3], dtype=tl.int64)
    dst_stride_3 = tl.cast(input=DST_STRIDES[3], dtype=tl.int64)
    src_stride_4 = tl.cast(input=SRC_STRIDES[4], dtype=tl.int64)
    dst_stride_4 = tl.cast(input=DST_STRIDES[4], dtype=tl.int64)
    src_stride_5 = tl.cast(input=SRC_STRIDES[5], dtype=tl.int64)
    dst_stride_5 = tl.cast(input=DST_STRIDES[5], dtype=tl.int64)
    # 循环级流水深度：允许编译器重叠相邻迭代的访存，与 launch 的 num_stages 选项不同。
    for block_start in tl.range(program_start, total_element_count, grid_stride, num_stages=NUM_STAGES):
        # 阶段二：全局索引拆成复制项编号和项内元素索引，再还原各维坐标。
        # 一个 block 可以跨越多对 Tensor；尾部 mask 同时保护指针表和数据访问。
        global_indices = block_start + element_lanes
        valid_elements = global_indices < total_element_count
        copy_index = global_indices // elements_per_copy
        element_indices = global_indices % elements_per_copy
        # 从最内层轴开始还原 6 维坐标；两端可以具有不同的物理 stride。
        axis5 = element_indices % SHAPE[5]
        remaining_indices = element_indices // SHAPE[5]
        axis4 = remaining_indices % SHAPE[4]
        remaining_indices = remaining_indices // SHAPE[4]
        axis3 = remaining_indices % SHAPE[3]
        remaining_indices = remaining_indices // SHAPE[3]
        axis2 = remaining_indices % SHAPE[2]
        remaining_indices = remaining_indices // SHAPE[2]
        axis1 = remaining_indices % SHAPE[1]
        remaining_indices = remaining_indices // SHAPE[1]
        axis0 = remaining_indices
        source_offsets = (
            axis0 * src_stride_0
            + axis1 * src_stride_1
            + axis2 * src_stride_2
            + axis3 * src_stride_3
            + axis4 * src_stride_4
            + axis5 * src_stride_5
        )
        destination_offsets = (
            axis0 * dst_stride_0
            + axis1 * dst_stride_1
            + axis2 * dst_stride_2
            + axis3 * dst_stride_3
            + axis4 * dst_stride_4
            + axis5 * dst_stride_5
        )

        # 阶段三：按元素所属复制项，从两张指针表读取源/目标地址，按元素宽度解释指针。
        # 使用等宽无符号类型保持原始位模式，不进行浮点数值转换。
        source_address = tl.load(pointer=SRC_POINTERS + copy_index, mask=valid_elements, other=0)
        destination_address = tl.load(pointer=DST_POINTERS + copy_index, mask=valid_elements, other=0)
        source_ptr = source_address.to(dtype=tl.pointer_type(element_ty=DTYPE))
        destination_ptr = destination_address.to(dtype=tl.pointer_type(element_ty=DTYPE))

        # 阶段四：读取源数据并写入目标，末尾越界元素不参与 load/store。
        values = tl.load(pointer=source_ptr + source_offsets, mask=valid_elements, other=0)
        tl.store(pointer=destination_ptr + destination_offsets, value=values, mask=valid_elements)

"""Full-attention DP 复制：显式传入各方案的存储，一次发射直接搬运完整 token。"""

import torch
import triton
import triton.language as tl

from ..utils import merge_contiguous_tail_dims, view_max_copy_dtype


def copy_gqa_tokens(
    src_k_storage: torch.Tensor,
    src_v_storage: torch.Tensor,
    dst_k_storage: torch.Tensor,
    dst_v_storage: torch.Tensor,
    src_token_indices: torch.Tensor,
    dst_token_indices: torch.Tensor,
) -> None:
    """融合 GQA/静态 FP8 的 K/V 复制；静态标定参数不参与搬运。"""
    _copy_token_storages(
        src_storages=(src_k_storage, src_v_storage),
        dst_storages=(dst_k_storage, dst_v_storage),
        src_token_indices=src_token_indices,
        dst_token_indices=dst_token_indices,
    )


def copy_group_quantized_tokens(
    src_k_storage: torch.Tensor,
    src_v_storage: torch.Tensor,
    src_k_scales: torch.Tensor,
    src_v_scales: torch.Tensor,
    dst_k_storage: torch.Tensor,
    dst_v_storage: torch.Tensor,
    dst_k_scales: torch.Tensor,
    dst_v_scales: torch.Tensor,
    src_token_indices: torch.Tensor,
    dst_token_indices: torch.Tensor,
) -> None:
    """融合 INT4/INT8 的 K/V payload 和动态 scale，分别保留字段 dtype。"""
    _copy_token_storages(
        src_storages=(src_k_storage, src_v_storage, src_k_scales, src_v_scales),
        dst_storages=(dst_k_storage, dst_v_storage, dst_k_scales, dst_v_scales),
        src_token_indices=src_token_indices,
        dst_token_indices=dst_token_indices,
    )


def copy_mla_tokens(
    src_storage: torch.Tensor,
    dst_storage: torch.Tensor,
    src_token_indices: torch.Tensor,
    dst_token_indices: torch.Tensor,
) -> None:
    """复制 MLA 完整行；packed 方案同时保留 RoPE、scale、indexer 和行 padding。"""
    _copy_token_storages(
        src_storages=(src_storage,),
        dst_storages=(dst_storage,),
        src_token_indices=src_token_indices,
        dst_token_indices=dst_token_indices,
    )


def _copy_token_storages(
    src_storages: tuple[torch.Tensor, ...],
    dst_storages: tuple[torch.Tensor, ...],
    src_token_indices: torch.Tensor,
    dst_token_indices: torch.Tensor,
) -> None:
    """共享的四维 token 布局搬运，不创建 payload 或索引的中间 Tensor。

    在目标设备当前流发射；调用方保证远端存储及索引已可通过 peer/IPC 访问，
    并负责源数据就绪、存储生命周期和完成事件。目标 token 不重复，源目标区域不重叠。
    """
    token_count = src_token_indices.numel()
    if token_count == 0:
        return

    # 阶段一：先合并两侧连续的 head/width，保留各自 layer/token 容量和 stride。
    # head 轴保留为 1，沿用四维 token kernel；不连续时保持原布局。
    # 再按 8/4/2 字节搬运，保留 FP8、NaN、动态 scale 等原始位模式。
    sources, destinations = [], []
    for source, destination in zip(src_storages, dst_storages):
        source, destination = merge_contiguous_tail_dims(
            copies=[(source, destination)],
            start_dim=-2,
            keepdim=True,
        )[0]
        source, destination = view_max_copy_dtype(copies=[(source, destination)])[0]
        sources.append(source)
        destinations.append(destination)

    # 阶段二：Tensor tuple 在编译时展开为独立指针参数，不上传指针表。
    # 一次 launch 完成当前方案全部字段；每个 program 用跳步循环覆盖任意 token 数。
    # 保持 24 个 program，使用 4096 个搬运单元提高跨卡读取吞吐，循环流水深度为 1。
    _copy_indexed_token_storages[(24,)](
        SOURCES=tuple(sources),
        DESTINATIONS=tuple(destinations),
        SRC_TOKEN_IDS=src_token_indices,
        DST_TOKEN_IDS=dst_token_indices,
        token_count=token_count,
        SRC_INDEX_STRIDE=src_token_indices.stride(0),
        DST_INDEX_STRIDE=dst_token_indices.stride(0),
        SHAPES=tuple((tensor.shape[0], tensor.shape[2], tensor.shape[3]) for tensor in sources),
        SRC_STRIDES=tuple(tensor.stride() for tensor in sources),
        DST_STRIDES=tuple(tensor.stride() for tensor in destinations),
        ELEMENT_BYTES=tuple(tensor.element_size() for tensor in sources),
        BLOCK_ELEMENTS=4096,
        NUM_STAGES=1,
        num_warps=8,
    )


@triton.jit
def _copy_indexed_token_storages(
    # 各字段的四维存储指针；K/V、scale 或 packed 行在一次 launch 中处理。
    SOURCES,
    DESTINATIONS,
    SRC_TOKEN_IDS,
    DST_TOKEN_IDS,
    token_count,
    # 索引 Tensor 可以非连续，直接使用调用方提供的 int32/int64 索引。
    SRC_INDEX_STRIDE: tl.constexpr,
    DST_INDEX_STRIDE: tl.constexpr,
    SHAPES: tl.constexpr,  # 各字段的 (layer_count, head_count, width)。
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    ELEMENT_BYTES: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    token_count = tl.cast(token_count, tl.int64)
    program_start = tl.program_id(0).to(tl.int64) * BLOCK_ELEMENTS
    grid_stride = tl.num_programs(0).to(tl.int64) * BLOCK_ELEMENTS
    lane = tl.arange(0, BLOCK_ELEMENTS).to(tl.int64)

    for field in tl.static_range(len(SHAPES)):
        # 每个字段独立计算地址；不同容量下 layer stride 不同，不能复用源端 stride。
        layers = SHAPES[field][0]
        heads = SHAPES[field][1]
        width = SHAPES[field][2]
        if ELEMENT_BYTES[field] == 8:
            copy_dtype = tl.uint64
        elif ELEMENT_BYTES[field] == 4:
            copy_dtype = tl.uint32
        elif ELEMENT_BYTES[field] == 2:
            copy_dtype = tl.uint16
        else:
            copy_dtype = tl.uint8
        source = SOURCES[field].to(tl.pointer_type(copy_dtype))
        destination = DESTINATIONS[field].to(tl.pointer_type(copy_dtype))
        element_count = token_count * layers * heads * width
        for block_start in tl.range(program_start, element_count, grid_stride, num_stages=NUM_STAGES):
            # 阶段一：按 [token, layer, head, width] 展开；除数均为布局常量，
            # 避免按动态 token_count 做逐元素的 int64 除法。
            element = block_start + lane
            valid = element < element_count
            column = element % width
            head = element // width % heads
            layer = element // (width * heads) % layers
            token = element // (width * heads * layers)

            # 阶段二：读取两端物理槽位；索引值及后续地址乘法使用 int64 防止溢出。
            src_token = tl.load(SRC_TOKEN_IDS + token * SRC_INDEX_STRIDE, mask=valid, other=0).to(tl.int64)
            dst_token = tl.load(DST_TOKEN_IDS + token * DST_INDEX_STRIDE, mask=valid, other=0).to(tl.int64)
            src_offset = (
                layer * tl.cast(SRC_STRIDES[field][0], tl.int64)
                + src_token * tl.cast(SRC_STRIDES[field][1], tl.int64)
                + head * tl.cast(SRC_STRIDES[field][2], tl.int64)
                + column * tl.cast(SRC_STRIDES[field][3], tl.int64)
            )
            dst_offset = (
                layer * tl.cast(DST_STRIDES[field][0], tl.int64)
                + dst_token * tl.cast(DST_STRIDES[field][1], tl.int64)
                + head * tl.cast(DST_STRIDES[field][2], tl.int64)
                + column * tl.cast(DST_STRIDES[field][3], tl.int64)
            )

            # 阶段三：直接从远端存储读出并写入目标槽位，无 gather 或跨设备中转存储。
            value = tl.load(source + src_offset, mask=valid, other=0)
            tl.store(destination + dst_offset, value, mask=valid)

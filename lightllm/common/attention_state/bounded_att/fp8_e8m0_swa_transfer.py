"""FP8/E8M0 SWA packed ring 与时间有序快照的 GPU 搬运。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from .fp8_e8m0_swa_state import FP8E8M0SlidingWindowConfig


def copy_swa_window(
    local: torch.Tensor,
    external: torch.Tensor,
    state_token_len: int,
    config: FP8E8M0SlidingWindowConfig,
    write: bool,
) -> None:
    """按绝对长度在 packed SWA ring 和 [layer, window, byte] 快照间双向复制。

    local 是选定请求的视图。external 可为已 pinned/registered 的大小页或 CUDA PD 页；
    直接由 GPU 读写，不分配 gather Tensor，不在主机等待拷贝完成。
    SWA 的数据和 scale 分区存储，不能直接按 token 行 memcpy。
    """
    elements = external.numel()
    _copy_swa_window[(min(24, triton.cdiv(elements, 1024)),)](
        LOCAL=local,
        EXTERNAL=external,
        window_start=state_token_len - external.shape[1],
        LOCAL_LAYER_STRIDE=local.stride(0),
        EXTERNAL_STRIDES=external.stride(),
        LAYERS=local.shape[0],
        RING_TOKENS=local.shape[1],
        WINDOW_TOKENS=external.shape[1],
        KV_BYTES=config.kv_bytes,
        KV_DATA_BYTES=config.kv_data_bytes,
        PAGE_TOKENS=config.page_token_size,
        PAGE_BYTES=config.get_packed_kv_bytes(config.page_token_size),
        WRITE=write,
        BLOCK=1024,
        num_warps=4,
    )


@triton.jit(do_not_specialize=["window_start"])
def _copy_swa_window(
    LOCAL,
    EXTERNAL,
    window_start,
    LOCAL_LAYER_STRIDE: tl.constexpr,
    EXTERNAL_STRIDES: tl.constexpr,
    LAYERS: tl.constexpr,
    RING_TOKENS: tl.constexpr,
    WINDOW_TOKENS: tl.constexpr,
    KV_BYTES: tl.constexpr,
    KV_DATA_BYTES: tl.constexpr,
    PAGE_TOKENS: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    WRITE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 阶段一：按线性快照拆出 layer/window/byte，绝对窗口起点保持运行时参数。
    lanes = tl.arange(0, BLOCK).to(tl.int64)
    count = LAYERS * WINDOW_TOKENS * KV_BYTES
    window_start = window_start.to(tl.int64)
    for start in range(tl.program_id(0).to(tl.int64) * BLOCK, count, tl.num_programs(0).to(tl.int64) * BLOCK):
        index = start + lanes
        byte = index % KV_BYTES
        window = index // KV_BYTES % WINDOW_TOKENS
        layer = index // (KV_BYTES * WINDOW_TOKENS)
        position = window_start + window
        slot = position % RING_TOKENS

        # 阶段二：SWA 每页先存全部 KV 数据，再存全部 scale，末尾保留物理 padding。
        packed_offset = slot // PAGE_TOKENS * PAGE_BYTES + tl.where(
            byte < KV_DATA_BYTES,
            slot % PAGE_TOKENS * KV_DATA_BYTES + byte,
            PAGE_TOKENS * KV_DATA_BYTES + slot % PAGE_TOKENS * (KV_BYTES - KV_DATA_BYTES) + byte - KV_DATA_BYTES,
        )
        local_offset = layer * tl.cast(LOCAL_LAYER_STRIDE, tl.int64) + packed_offset
        external_offset = (
            layer * tl.cast(EXTERNAL_STRIDES[0], tl.int64)
            + window * tl.cast(EXTERNAL_STRIDES[1], tl.int64)
            + byte * tl.cast(EXTERNAL_STRIDES[2], tl.int64)
        )
        valid = index < count
        has_history = valid & (position >= 0)
        # TODO：按数据/scale 两个分区扩大搬运单位，融合宽类型访问，保持 packed padding 不变。
        if WRITE:
            values = tl.load(LOCAL + local_offset, mask=has_history, other=0)
            tl.store(EXTERNAL + external_offset, values, mask=valid)
        else:
            values = tl.load(EXTERNAL + external_offset, mask=has_history, other=0)
            tl.store(LOCAL + local_offset, values, mask=has_history)

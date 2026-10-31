"""窗口压缩器候选/gate ring 与时间有序快照的 GPU 搬运。"""

import torch
import triton

from ..utils import view_max_copy_dtype
from ..att_tail.kpool_transfer import _transfer_kpool_pages


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
    _transfer_kpool_pages[(min(24, triton.cdiv(elements, 1024)),)](
        LOCAL=local_view,
        EXTERNAL=external_view,
        local_window_start=(state_token_len - window) % local.shape[1],
        external_window_start=0,
        invalid_prefix=max(window - state_token_len, 0),
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

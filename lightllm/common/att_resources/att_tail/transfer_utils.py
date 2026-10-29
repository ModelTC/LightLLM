"""Attention tail 的 PD 页传输：在本地 ring 和固定逻辑窗口之间搬运。"""

import torch

from ..types import StateTransferLayout


def att_tail_buffer_copies(
    tail: torch.Tensor,
    page_buffers: torch.Tensor,
    layout: StateTransferLayout,
    *,
    page_index: int,
    state_token_len: int,
    write: bool,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """按绝对位置映射 ring 与逻辑窗口，短前缀写出时将不存在的位置补零。

    tail 轴为 (layer, ring_slot, channel)，外部页轴为 (layer, window, channel)。
    window 按时间顺序对应 [state_token_len - pool_size, state_token_len)，
    本地物理槽位使用绝对位置对 tail.shape[1] 取模，因此不要求 P/D 的 MTP 步数相同。
    返回有效位置的 (源视图, 目标视图)，由 State 提交复制；读回前由 State 清零本地 ring。
    """
    assert page_buffers.ndim >= 2, "page buffers must include a leading page dimension"
    page_bytes = page_buffers[page_index].view(torch.uint8).view(-1)
    copies = []
    for region in layout.regions:
        assert region.field == "tail", f"unknown transfer field {region.field}"
        external = page_bytes[region.byte_start:region.byte_end].view(region.dtype).view(region.shape)
        external = external.permute(tuple(region.axes.index(axis) for axis in ("layer", "window", "channel")))
        assert tail.dtype == external.dtype, "local and external field dtypes must match"
        assert (tail.shape[0], tail.shape[2]) == (external.shape[0], external.shape[2]), (
            "local and external tail layer/channel shapes must match"
        )
        if write and not region.write_enabled:
            continue
        # TODO：用融合传输 kernel 完成逻辑窗口与 ring 的取模映射、写出补零和数据搬运，
        # 减少逐窗口槽位 copy_/fill_；必须保留 P/D 不同 MTP 步数及 TP 副本只由 rank 0 写出的语义。
        # 请求级低频搬运；逐槽位取视图，不创建索引 Tensor，也不读取 GPU 上的游标。
        window_start = state_token_len - external.shape[1]
        for window_index in range(external.shape[1]):
            position = window_start + window_index
            external_view = external[:, window_index]
            if position < 0:
                if write:
                    external_view.fill_(0)
                continue
            local_view = tail[:, position % tail.shape[1]]
            copies.append((local_view, external_view) if write else (external_view, local_view))
    return copies

from __future__ import annotations

import torch

from .token_storage import TokenStorage


class TokenTransfer:
    """按规格搬运编码字节，不转换精度、不发布缓存或分配目标槽位。

    页面字段的 Tensor 和轴绑定为一个 TensorField；源数据须已就绪。
    所有操作提交到当前 CUDA 流，调用方负责等待完成及保留两端存储和索引。
    """

    def __init__(self, storage: TokenStorage):
        self.storage = storage

    def copy_tokens(self, src_indices: torch.Tensor, dst_indices: torch.Tensor) -> None:
        assert src_indices.is_cuda and dst_indices.is_cuda, "token indices must reside on CUDA"
        assert (
            src_indices.ndim == dst_indices.ndim == 1 and src_indices.numel() == dst_indices.numel()
        ), "token copies require equal one-dimensional index tensors"
        assert src_indices.dtype in (torch.int32, torch.int64) and dst_indices.dtype in (
            torch.int32,
            torch.int64,
        ), "token indices must be integer tensors"
        assert self.storage.buffers and all(
            buffer.is_cuda for buffer in self.storage.buffers.values()
        ), "token copies require CUDA storage"
        assert (
            src_indices.numel() % self.storage.spec.block_tokens == 0
        ), "token copies require complete compressed blocks"
        if not src_indices.numel():
            return
        from lightllm.common.basemodel.triton_kernel.kv_move import copy_kv_buffer_to_kv_buffer

        for buffer in self.storage.buffers.values():
            copy_kv_buffer_to_kv_buffer(src_mem_index=src_indices, dst_mem_index=dst_indices, kv_buffer=buffer)

    def copy_from_peer(self, source: TokenStorage, src_indices, dst_indices) -> None:
        """复制相同本地布局的 peer 存储；模型/静态标定一致性须由连接方另行确认。"""
        assert source.spec == self.storage.spec, "DP copy requires equivalent storage specifications"
        assert src_indices.is_cuda and dst_indices.is_cuda, "DP token indices must reside on CUDA"
        assert src_indices.numel() == dst_indices.numel(), "source and destination token counts differ"
        assert (
            src_indices.numel() % self.storage.spec.block_tokens == 0
        ), "DP token copies require complete compressed blocks"
        from .dp_transfer import copy_token_storages

        copy_token_storages(
            src_storages=tuple(source.buffers[field.name] for field in source.spec.fields),
            dst_storages=tuple(self.storage.buffers[field.name] for field in self.storage.spec.fields),
            src_token_indices=src_indices,
            dst_token_indices=dst_indices,
        )

    def pages(self, token_indices, page_buffers, page_indices, layout, write, page_readies=None) -> None:
        from .page_transfer import transfer_token_pages

        transfer_token_pages(
            fields=self.storage.spec.page_views(self.storage.buffers),
            token_indices=token_indices,
            page_buffers=page_buffers,
            page_indices=page_indices,
            layout=layout,
            write=write,
            page_readies=page_readies,
        )


class BlockTokenTransfer(TokenTransfer):
    """以完整原始 token 块搬运压缩数据；PD 补齐的尾部槽位也须有有效存储支撑。"""

    def pages(self, token_indices, page_buffers, page_indices, layout, write, page_readies=None):
        alignment = self.storage.spec.block_tokens
        if page_buffers.device.type == "cpu":
            page_tokens = layout.tokens_per_page
            assert (
                page_tokens is not None and page_tokens > 0 and page_tokens % alignment == 0
            ), "CPU page token capacity must be a positive multiple of block_token_size"
            from math import prod

            page_bytes = prod(page_buffers.shape[1:]) * page_buffers.element_size()
            required_bytes = self.storage.spec.token_bytes * page_tokens
            assert (
                0 <= layout.byte_start < layout.byte_start + required_bytes <= layout.byte_end <= page_bytes
            ), "compressed State byte range must fit within one CPU page"
            assert all(0 <= page < page_buffers.shape[0] for page in page_indices), "CPU page index is out of bounds"
        else:
            padding = (-token_indices.numel()) % alignment
            if token_indices.numel() and padding:
                tail = token_indices[-1:] + torch.arange(
                    start=1, end=padding + 1, dtype=token_indices.dtype, device=token_indices.device
                )
                token_indices = torch.cat(tensors=(token_indices, tail))
        super().pages(token_indices, page_buffers, page_indices, layout, write, page_readies)

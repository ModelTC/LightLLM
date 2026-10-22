from types import SimpleNamespace

import torch

from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextLinearAttPageHelper
from lightllm.common.linear_state import LinearStateManager
from lightllm.common.state_cache_manager import LayerCache


class LinearAttPDState:
    """在线性请求状态与 PD 块之间拷贝，复用已有页面布局和 TP 分片逻辑。

    内部页面 helper 只持有静态配置，GPU 状态仍由 LinearStateManager 管理。
    不持有请求管理器、内存管理器或 CPU 缓存池，也不负责流同步。
    """

    def __init__(self, linear_state_manager: LinearStateManager, indexer_tail: LayerCache = None):
        self.linear_state_manager = linear_state_manager
        self.linear_config = linear_state_manager.linear_config
        self.indexer_tail = indexer_tail
        self._page_helper = Qwen3NextLinearAttPageHelper(SimpleNamespace(linear_config=self.linear_config))
        self.state_nbytes = self._page_helper.state_nbytes
        if self.indexer_tail is not None:
            self.tail_dtype = self.indexer_tail.buffer.dtype
            self.tail_shape = (
                self.linear_config.get_full_att_kv_layer_num_with_draft_model(),
                *self.indexer_tail.buffer.shape[2:],
            )
            self.tail_offset = ((self.state_nbytes + 15) // 16) * 16
            self.tail_nbytes = self.tail_shape[0] * self.tail_shape[1] * self.tail_shape[2] * self.tail_dtype.itemsize
            self.state_nbytes = self.tail_offset + self.tail_nbytes

    def _view_block_bytes(self, state_block: torch.Tensor):
        assert state_block.is_contiguous(), "PD state block must be contiguous"
        block_nbytes = state_block.numel() * state_block.element_size()
        assert (
            block_nbytes >= self.state_nbytes
        ), f"PD state block bytes {block_nbytes} is smaller than global linear att state bytes {self.state_nbytes}"
        return state_block.reshape(-1).view(torch.uint8)

    def _view_block_to_linear_att_state(self, state_block: torch.Tensor):
        block_bytes = self._view_block_bytes(state_block)
        helper = self._page_helper
        conv = block_bytes[: helper.conv_nbytes].view(self.linear_config.conv_state_dtype).view(helper.conv_shape)
        ssm = (
            block_bytes[helper.ssm_offset : helper.ssm_offset + helper.ssm_nbytes]
            .view(self.linear_config.ssm_state_dtype)
            .view(helper.ssm_shape)
        )
        return conv, ssm

    def _view_block_to_tail(self, state_block: torch.Tensor):
        block_bytes = self._view_block_bytes(state_block)
        return (
            block_bytes[self.tail_offset : self.tail_offset + self.tail_nbytes]
            .view(self.tail_dtype)
            .view(self.tail_shape)
        )

    def _check_tp_rank(self, tp_rank: int):
        assert (
            isinstance(tp_rank, int) and 0 <= tp_rank < self.linear_config.tp_world_size
        ), f"invalid TP rank {tp_rank} for world size {self.linear_config.tp_world_size}"

    def get_pd_state(self, req_idx: int, state_block: torch.Tensor, tp_rank: int):
        """将本 rank 已接受的请求状态写入块中对应分片，保持请求运行态不变。

        调用方负责汇总所有 TP rank，并在源请求推进或释放前完成传输拷贝。
        GLM indexer tail 是 TP 副本，仅由 rank 0 写入。
        """
        self._check_tp_rank(tp_rank)
        conv_page, ssm_page = self._view_block_to_linear_att_state(state_block)
        tail_page = self._view_block_to_tail(state_block) if self.indexer_tail is not None else None
        self._page_helper._write_one_rank(self, tp_rank, req_idx, conv_page, ssm_page)
        if tail_page is not None and tp_rank == 0:
            tail_page.copy_(self.indexer_tail.buffer[:, req_idx], non_blocking=True)

    def load_pd_state(self, req_idx: int, state_block: torch.Tensor, tp_rank: int):
        """解析全局传输块，将本 rank 分片写回请求并重置旧的 MTP/replay 跟踪。

        拷贝在当前 CUDA stream 上执行，调用方通过 PD 完成事件保证后续推理可见。
        """
        self._check_tp_rank(tp_rank)
        conv_page, ssm_page = self._view_block_to_linear_att_state(state_block)
        tail_page = self._view_block_to_tail(state_block) if self.indexer_tail is not None else None
        self._page_helper._read_one_rank(self, tp_rank, req_idx, conv_page, ssm_page)
        if tail_page is not None:
            self.indexer_tail.buffer[:, req_idx].copy_(tail_page, non_blocking=True)

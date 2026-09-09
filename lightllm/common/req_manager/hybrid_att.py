from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import torch

from .base import ReqManager


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class HybridAttentionReqManager(ReqManager, ABC):
    """混合 attention 的请求运行态与大小页 checkpoint 管理接口。

    大小页沿同一虚拟 token 索引空间匹配前缀，full attention KV 保持 token 粒度存储。
    linear/sliding-window 状态在大页边界及请求可缓存尾部的小页边界保存 checkpoint，
    缓存命中后，再将相应 checkpoint 恢复到请求运行态。

    公共缓存流程负责大小页分配、边界、匹配与淘汰；各实现负责状态存储和保存/恢复。
    """

    @abstractmethod
    def create_state_cache_manager(self, size: int):
        """Return checkpoint storage used by request-state page boundaries."""

    @abstractmethod
    def init_hybrid_attention_state(self, req: "InferReq"):
        """Initialize request runtime state when no prefix cache is restored."""

    def restore_big_page_state(self, big_page_buffer_idx: int, req: "InferReq"):
        self.restore_state(req, self.mem_manager.linear_att_big_page_buffers, big_page_buffer_idx)

    def restore_small_page_state(self, req: "InferReq", small_page_buffers):
        self.restore_state(req, small_page_buffers, req.shared_kv_node.small_page_buffer_idx)

    @abstractmethod
    def restore_state(self, req: "InferReq", state_cache_manager, buffer_idx: int):
        """Restore the same request-state payload from either checkpoint pool."""

    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        """Default checkpoint copies; models may override with a batched kernel."""
        for req_idx, buffer_idx in zip(req_indexes, buffer_indexes):
            if buffer_idx != -1:
                self.save_state(req_idx, buffer_idx, self.mem_manager.linear_att_big_page_buffers)

    @abstractmethod
    def save_state(self, req_idx: int, buffer_idx: int, state_cache_manager):
        """Save a request's payload into either checkpoint pool."""

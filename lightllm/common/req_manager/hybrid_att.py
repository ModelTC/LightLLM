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

    @abstractmethod
    def restore_big_page_state(self, big_page_buffer_idx: int, req: "InferReq"):
        """Restore runtime state from a big-page checkpoint."""

    @abstractmethod
    def restore_small_page_state(self, req: "InferReq", small_page_buffers):
        """Restore runtime state from a small-page checkpoint."""

    @abstractmethod
    def save_big_page_states(self, b_req_idx: torch.Tensor, req_indexes: List[int], buffer_indexes: List[int]):
        """Save selected checkpoints; CPU request IDs avoid device-to-host synchronization."""

    @abstractmethod
    def save_small_page_state(self, req_idx: int, buffer_idx: int, small_page_buffers):
        """Save a request's final small-page checkpoint in the layout's storage."""

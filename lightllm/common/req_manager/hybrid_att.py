from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import torch

from .base import ReqManager


if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


class HybridAttentionReqManager(ReqManager, ABC):
    """Request manager contract for token/full + request-state attention models.

    The token index table remains the virtual, token-granular address space used
    by prefix-cache matching.  The non-full attention state is managed through
    this interface and may have a different physical granularity.
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

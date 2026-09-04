from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Union

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

    is_linear_attention = False

    @abstractmethod
    def create_state_cache_manager(self, size: int):
        """Create checkpoint storage used by request-state page boundaries."""

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
    def copy_runtime_state_to_cache(
        self,
        req_indexes: Union[List[int], torch.Tensor],
        buffer_indexes: List[int],
        state_cache_manager,
    ):
        """Copy selected request runtime states into host-side page buffers."""

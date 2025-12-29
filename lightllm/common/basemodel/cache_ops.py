"""
Cache operation protocols for model-specific KV cache management.

This module defines interfaces for models with custom buffer management needs
(e.g., multi-buffer KV cache, custom synchronization requirements).

The protocols use Python's Protocol type with runtime_checkable to enable
isinstance() checks for graceful degradation. Models can opt-in to custom
cache management by implementing these protocols without requiring inheritance.
"""

from typing import Optional, List, Protocol, runtime_checkable
import torch


@runtime_checkable
class MultiBufferMemoryManager(Protocol):
    """
    Protocol for memory managers that manage multiple KV cache buffers.

    Models like DeepSeek V3.2 (NSA) maintain additional buffers beyond the
    standard KV cache (e.g., indexer_ks buffer for top-k selection).

    This protocol provides a generic interface for accessing and manipulating
    these auxiliary buffers without the common module needing to know about
    model-specific implementations.
    """

    def get_aux_buffer_names(self) -> List[str]:
        """
        Return names of auxiliary buffers managed by this memory manager.

        Example: ["indexer_ks", "another_buffer"]

        Returns:
            List of buffer names
        """
        ...

    def get_aux_buffer(self, name: str) -> Optional[torch.Tensor]:
        """
        Get auxiliary buffer by name.

        Args:
            name: Name of the buffer

        Returns:
            Buffer tensor or None if not found
        """
        ...

    def copy_aux_buffer_tokens(
        self,
        buffer_name: str,
        layer_idx: int,
        src_positions: torch.Tensor,
        dest_positions: torch.Tensor,
    ) -> None:
        """
        Copy tokens in auxiliary buffer from source to destination positions.

        Called during prefix cache operations to synchronize auxiliary buffers
        with the main KV cache.

        Args:
            buffer_name: Name of auxiliary buffer (e.g., "indexer_ks")
            layer_idx: Layer index
            src_positions: Source token positions [N]
            dest_positions: Destination token positions [N]
        """
        ...


@runtime_checkable
class PrefillHookProvider(Protocol):
    """
    Protocol for models that need custom prefill behavior.

    Models can implement hooks to inject custom logic into the prefill
    lifecycle without modifying the base class. This enables:

    1. Capturing state before KV cache positions are updated
    2. Synchronizing model-specific buffers after KV cache update

    Example use case: DeepSeek V3.2 needs to synchronize its indexer_ks buffer
    when prefix cache is hit.
    """

    def capture_prefill_state(
        self,
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
    ) -> object:
        """
        Capture state before KV cache positions are updated.

        Called before init_req_to_token_indexes() updates memory positions.
        The returned state is passed to sync_prefill_buffers().

        Example: DeepSeek V3.2 captures old indexer_ks positions.

        Args:
            req_to_token_indexs: Request to token index mapping
            b_req_idx: Batch request indices
            b_ready_cache_len: Batch ready cache lengths (for prefix cache)

        Returns:
            Captured state (passed to sync_prefill_buffers)
        """
        ...

    def sync_prefill_buffers(
        self,
        captured_state: object,
        req_to_token_indexs: torch.Tensor,
        b_req_idx: torch.Tensor,
        b_ready_cache_len: torch.Tensor,
    ) -> None:
        """
        Synchronize model-specific buffers after KV cache update.

        Called after init_req_to_token_indexes() has reorganized the KV cache.
        Use the captured_state to restore auxiliary buffer consistency.

        Example: DeepSeek V3.2 copies indexer_ks to new positions.

        Args:
            captured_state: State returned from capture_prefill_state()
            req_to_token_indexs: Request to token index mapping (new positions)
            b_req_idx: Batch request indices
            b_ready_cache_len: Batch ready cache lengths
        """
        ...


def has_prefill_hooks(model) -> bool:
    """
    Check if model provides prefill hooks.

    Uses isinstance() check with Protocol for duck-typing.

    Args:
        model: Model instance to check

    Returns:
        True if model implements PrefillHookProvider protocol
    """
    return isinstance(model, PrefillHookProvider)


def is_multi_buffer_model(mem_manager) -> bool:
    """
    Check if model has multi-buffer memory manager.

    Args:
        mem_manager: Memory manager to check

    Returns:
        True if memory manager implements MultiBufferMemoryManager protocol
    """
    return isinstance(mem_manager, MultiBufferMemoryManager)

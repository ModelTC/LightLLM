"""
Helper utilities for cache operations.

Provides reusable components for models implementing multi-buffer cache management.
These utilities reduce code duplication and provide a consistent pattern for
future models with similar needs.
"""

from typing import Optional, List, Callable
import torch


def capture_old_positions(
    req_to_token_indexs: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
) -> List[Optional[torch.Tensor]]:
    """
    Generic helper to capture old positions before cache reorganization.

    This function can be reused by any model that needs to track old positions
    for auxiliary buffer synchronization. It captures positions for tokens
    that have ready cache (i.e., prefix cache hits).

    Args:
        req_to_token_indexs: Request to token index mapping [max_req, max_len]
        b_req_idx: Batch request indices [batch_size]
        b_ready_cache_len: Batch ready cache lengths [batch_size]

    Returns:
        List of length batch_size where each element is either:
        - None (no cached tokens for this request)
        - Tensor of old positions for cached tokens

    Example:
        >>> req_to_token_indexs = torch.arange(1000).view(100, 10)
        >>> b_req_idx = torch.tensor([0, 1, 2])
        >>> b_ready_cache_len = torch.tensor([5, 0, 3])
        >>> old_positions = capture_old_positions(
        ...     req_to_token_indexs, b_req_idx, b_ready_cache_len
        ... )
        >>> len(old_positions)
        3
        >>> old_positions[0].shape[0]
        5
        >>> old_positions[1] is None
        True
    """
    old_positions = []
    for i in range(b_req_idx.shape[0]):
        req_idx = b_req_idx[i].item()
        ready_cache_len = b_ready_cache_len[i].item()

        if ready_cache_len > 0:
            # Clone old positions for cached tokens
            old_pos = req_to_token_indexs[req_idx, 0:ready_cache_len].clone()
            old_positions.append(old_pos)
        else:
            # No cached tokens for this request
            old_positions.append(None)

    return old_positions


def copy_buffer_tokens(
    buffer,
    layer_idx: int,
    b_req_idx: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    old_positions: List[Optional[torch.Tensor]],
    copy_fn: Callable,
) -> None:
    """
    Generic helper to copy auxiliary buffer tokens.

    This function handles the logic of iterating through requests and
    copying tokens from old positions to new positions. It works with
    both per-layer buffers and unified buffers.

    Args:
        buffer: Auxiliary buffer (can be multi-layer list or single tensor)
        layer_idx: Layer index
        b_req_idx: Batch request indices [batch_size]
        b_ready_cache_len: Batch ready cache lengths [batch_size]
        req_to_token_indexs: Request to token index mapping (new positions)
        old_positions: List of old positions per request
        copy_fn: Function to perform actual copy: copy_fn(buffer, src, dest)

    Example:
        >>> def my_copy_fn(buffer, src, dest):
        ...     # Custom copy logic
        ...     buffer[dest] = buffer[src]
        >>> copy_buffer_tokens(
        ...     buffer=my_buffer,
        ...     layer_idx=0,
        ...     b_req_idx=torch.tensor([0, 1]),
        ...     b_ready_cache_len=torch.tensor([5, 3]),
        ...     req_to_token_indexs=new_positions,
        ...     old_positions=old_pos_list,
        ...     copy_fn=my_copy_fn,
        ... )
    """
    for i in range(b_req_idx.shape[0]):
        req_idx = b_req_idx[i].item()
        ready_cache_len = b_ready_cache_len[i].item()
        old_pos = old_positions[i]

        if ready_cache_len > 0 and old_pos is not None:
            # New positions after KV cache reorganization
            new_pos = req_to_token_indexs[req_idx, 0:ready_cache_len]

            # Handle both per-layer and unified buffers
            if isinstance(buffer, list):
                layer_buffer = buffer[layer_idx]
            else:
                layer_buffer = buffer

            # Perform the copy
            copy_fn(layer_buffer, old_pos, new_pos)

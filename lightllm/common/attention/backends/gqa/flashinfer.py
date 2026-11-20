from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from . import GQAInputs

if TYPE_CHECKING:
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


@dataclass(frozen=True)
class FlashInferPrefillBatch:
    """Borrow a planned NHD wrapper for packed, causal prefill.

    The caller plans query offsets, page indices and last-page lengths for
    the current batch, with scale 1/sqrt(head_dim) and no custom mask, sliding
    window, positional encoding or soft cap. Planned dtypes and head counts
    must match the operands. Page index p addresses cache slots
    [p * page_size, (p + 1) * page_size).

    The caller owns the wrapper, metadata and mutable workspace and keeps
    them alive through execution. Planning, reuse and graph lifetimes remain
    with that owner, which must serialize use of shared workspace. The
    backend neither plans nor retains the wrapper.
    """

    wrapper: "BatchPrefillWithPagedKVCacheWrapper"
    page_size: int


@dataclass(frozen=True)
class FlashInferDecodeBatch:
    """Borrow a planned NHD tensor-core wrapper for one query per request.

    Planning and storage follow FlashInferPrefillBatch's contract. For CUDA
    graph replay, the caller refreshes the captured wrapper's plan and
    metadata in place, preserving the captured workspace and buffer addresses.
    A temporary replay state without a wrapper is not an execution batch.
    """

    wrapper: "BatchDecodeWithPagedKVCacheWrapper"
    page_size: int


class FlashInferGQAPrefillBackend(AttentionBackend[FlashInferPrefillBatch, GQAInputs]):
    def forward(
        self,
        batch: FlashInferPrefillBatch,
        inputs: GQAInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape, dtype and device.

        Q/out are contiguous FP16/BF16 tensors on the same CUDA device as
        the cache and plan. Cache slots must form complete pages; head_dim
        and head strides are contiguous, while the slot stride may include
        padding (including K/V slices of shared storage). Q/K/V are read-only;
        out must not overlap them or the plan's metadata/workspace.
        Only sliding_window=(-1, -1), the full causal prefix, is supported.
        Scratch is already owned by the planned wrapper, so forward does not
        call alloc_func.
        """
        assert inputs.sliding_window == (-1, -1), "FlashInfer sliding_window is not supported"
        batch.wrapper.run(
            q=inputs.q,
            paged_kv_cache=(
                inputs.k_cache.view(-1, batch.page_size, inputs.k_cache.shape[1], inputs.k_cache.shape[2]),
                inputs.v_cache.view(-1, batch.page_size, inputs.v_cache.shape[1], inputs.v_cache.shape[2]),
            ),
            out=out,
        )
        return out


class FlashInferGQADecodeBackend(AttentionBackend[FlashInferDecodeBatch, GQAInputs]):
    def forward(
        self,
        batch: FlashInferDecodeBatch,
        inputs: GQAInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q's shape, dtype and device.

        Tensor layout, mutation, allocation and sliding_window rules match
        the prefill backend, with one query per request in the planned batch
        order.
        """
        assert inputs.sliding_window == (-1, -1), "FlashInfer sliding_window is not supported"
        batch.wrapper.run(
            q=inputs.q,
            paged_kv_cache=(
                inputs.k_cache.view(-1, batch.page_size, inputs.k_cache.shape[1], inputs.k_cache.shape[2]),
                inputs.v_cache.view(-1, batch.page_size, inputs.v_cache.shape[1], inputs.v_cache.shape[2]),
            ),
            out=out,
        )
        return out

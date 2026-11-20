from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta


@dataclass(frozen=True)
class FlashMLASparsePrefillBatch:
    # Indices address the rows of this layer's compact KV, including any cached prefix.
    indices: torch.Tensor


@dataclass(frozen=True)
class SparseMLAPrefillInputs:
    """q [tokens, heads, 576] and compact kv [rows, 1, 576] are borrowed BF16 operands."""

    q: torch.Tensor
    kv: torch.Tensor
    softmax_scale: float


class FlashMLASparsePrefillBackend(AttentionBackend[FlashMLASparsePrefillBatch, SparseMLAPrefillInputs]):
    def forward(
        self,
        batch: FlashMLASparsePrefillBatch,
        inputs: SparseMLAPrefillInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Write to out [tokens, heads, 512] on q.device and return it. Causality is encoded in indices.

        Each int32 index selects one KV row for that query (-1 is padding). out is BF16
        with unit last stride and must be disjoint from the inputs and indices.
        """
        from flash_mla import flash_mla_sparse_fwd

        indices = batch.indices
        flash_mla_sparse_fwd(
            q=inputs.q,
            kv=inputs.kv,
            indices=indices.unsqueeze(1) if indices.ndim == 2 else indices,
            sm_scale=inputs.softmax_scale,
            d_v=out.shape[-1],
            out=out,
        )
        return out


@dataclass(frozen=True)
class FlashMLASparseDecodeBatch:
    # Physical FP8 cache slots, one top-k row per query. -1 marks an unused slot.
    indices: torch.Tensor
    # Caller-owned. FlashMLA initializes this on first use for the batch geometry.
    sched_meta: "FlashMLASchedMeta"


@dataclass(frozen=True)
class SparseMLADecodeInputs:
    """BF16 absorbed and rotary queries, plus a borrowed per-layer packed FP8 cache.

    Each physical row has 512 latent FP8 bytes, four FP32 scales, and 64 BF16 rotary values.
    The row stride may include padding. Cache population belongs to the caller.
    """

    q_nope: torch.Tensor
    q_rope: torch.Tensor
    packed_kv: torch.Tensor
    softmax_scale: float


class FlashMLASparseDecodeBackend(AttentionBackend[FlashMLASparseDecodeBatch, SparseMLADecodeInputs]):
    def forward(
        self,
        batch: FlashMLASparseDecodeBatch,
        inputs: SparseMLADecodeInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Attend to selected physical slots, one query per request. Fill and return out.

        out is contiguous BF16 [requests, heads, 512] on q_nope.device, disjoint from operands and metadata.
        """
        from flash_mla import flash_mla_with_kvcache

        indices = batch.indices
        if indices.ndim == 2:
            indices = indices.unsqueeze(1)
        assert indices.shape[1] == 1, "FlashMLA sparse decode expects one query per request"
        q_nope, q_rope = inputs.q_nope, inputs.q_rope
        q = alloc_func(
            (*q_nope.shape[:-1], q_nope.shape[-1] + q_rope.shape[-1]), dtype=q_nope.dtype, device=q_nope.device
        )
        torch.cat((q_nope, q_rope), dim=-1, out=q)
        packed = inputs.packed_kv
        # FlashMLA expects singleton page/head strides to match the packed row width.
        kv = torch.as_strided(
            packed,
            size=(packed.shape[0], 1, 1, packed.shape[-1]),
            stride=(packed.stride(0), packed.shape[-1], packed.shape[-1], packed.stride(-1)),
        )
        flash_mla_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=kv,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=out.shape[-1],
            tile_scheduler_metadata=batch.sched_meta,
            softmax_scale=inputs.softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            out=out.unsqueeze(1),
        )
        return out

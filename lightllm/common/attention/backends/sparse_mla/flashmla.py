from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta


@dataclass(frozen=True)
class FlashMLASparsePrefillBatch:
    """Sparse prefill has no state shared across layers; each layer supplies its own indices."""


@dataclass(frozen=True)
class SparseMLAPrefillInputs:
    """q [tokens, heads, 576] and compact kv [rows, 1, 576] are borrowed BF16 operands.

    indices [tokens, topk] or [tokens, 1, topk] are int32 rows of this layer's compact kv,
    including any cached prefix; -1 is padding.
    """

    q: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    softmax_scale: float


class FlashMLASparsePrefillBackend(AttentionBackend[FlashMLASparsePrefillBatch, SparseMLAPrefillInputs]):
    def plan(self) -> FlashMLASparsePrefillBatch:
        return FlashMLASparsePrefillBatch()

    def forward(
        self,
        batch: FlashMLASparsePrefillBatch,
        inputs: SparseMLAPrefillInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Write to out [tokens, heads, 512] on q.device and return it. Causality is encoded in indices.

        out is BF16 with unit last stride and must be disjoint from the inputs.
        """
        from flash_mla import flash_mla_sparse_fwd

        indices = inputs.indices
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
    # FlashMLA fills this tile schedule on the first forward of the pass and every later layer reuses it.
    sched_meta: "FlashMLASchedMeta"


@dataclass(frozen=True)
class SparseMLADecodeInputs:
    """BF16 absorbed and rotary queries, plus a borrowed per-layer packed FP8 cache.

    Each physical row has 512 latent FP8 bytes, four FP32 scales, and 64 BF16 rotary values.
    The row stride may include padding. Cache population belongs to the caller.
    indices [requests, topk] or [requests, 1, topk] are int32 physical cache slots chosen
    for this layer; -1 marks an unused slot.
    """

    q_nope: torch.Tensor
    q_rope: torch.Tensor
    packed_kv: torch.Tensor
    indices: torch.Tensor
    softmax_scale: float


class FlashMLASparseDecodeBackend(AttentionBackend[FlashMLASparseDecodeBatch, SparseMLADecodeInputs]):
    def plan(self) -> FlashMLASparseDecodeBatch:
        """Create an empty tile schedule. Under CUDA graphs it is filled before capture and reused by replay."""
        from flash_mla import get_mla_metadata

        sched_meta, _ = get_mla_metadata()
        return FlashMLASparseDecodeBatch(sched_meta=sched_meta)

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

        indices = inputs.indices
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

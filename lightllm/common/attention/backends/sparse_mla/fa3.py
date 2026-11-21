"""FA3 sparse MLA decode: the top-k cache slots serve as a one-token-per-page page table."""

from dataclasses import dataclass

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBatch,
    AttentionImpl,
    AttentionStepBuilder,
    TensorAllocator,
)
from .. import keep_for_cuda_graph
from . import SparseMLADecodeLayerInputs, SparseMLALayerConfig


@dataclass(frozen=True)
class FA3SparseMLADecodeStepArgs:
    """cache_seqlens is min(topk, b_seq_len) per request, the number of selected slots it reads;
    cu_seqlens_k is its int32 prefix sum and cu_seqlens_q the int32 query offsets."""

    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int


class FA3SparseMLADecodeStepBuilder(AttentionStepBuilder[FA3SparseMLADecodeStepArgs]):
    def __init__(self, *, topk: int) -> None:
        """topk is the indexer's selection width."""
        self.topk = topk
        self._cuda_graph_tensors: dict[tuple, torch.Tensor] = {}

    def build(self, batch: AttentionBatch) -> FA3SparseMLADecodeStepArgs:
        """A CUDA graph step keeps the length tensors per (microbatch, batch size), rewritten in place."""
        device = batch.b_seq_len.device
        cache_seqlens = torch.minimum(
            torch.full((batch.batch_size,), self.topk, dtype=torch.int32, device=device), batch.b_seq_len
        )
        padded = torch.zeros(batch.batch_size + 1, dtype=torch.int32, device=device)
        padded[1:].copy_(cache_seqlens, non_blocking=True)
        cu_seqlens_k = padded.cumsum(dim=0, dtype=torch.int32)
        if batch.uses_cuda_graph:
            key, store = (batch.microbatch_index, batch.batch_size), self._cuda_graph_tensors
            cache_seqlens = keep_for_cuda_graph(store, (*key, "cache_seqlens"), cache_seqlens)
            cu_seqlens_k = keep_for_cuda_graph(store, (*key, "cu_seqlens_k"), cu_seqlens_k)
        return FA3SparseMLADecodeStepArgs(
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=batch.cu_q_seq_len,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=batch.max_q_seq_len,
        )


class FA3SparseMLADecodeImpl(
    AttentionImpl[SparseMLALayerConfig, FA3SparseMLADecodeStepArgs, SparseMLADecodeLayerInputs]
):
    def __init__(self, layer_config: SparseMLALayerConfig) -> None:
        super().__init__(layer_config)
        if layer_config.kv_cache_format != "bf16":
            raise NotImplementedError("FA3 sparse MLA decode supports only the bf16 cache")

    def forward(
        self,
        step_args: FA3SparseMLADecodeStepArgs,
        layer_inputs: SparseMLADecodeLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q_nope's shape."""
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        layer_config = self.layer_config
        assert layer_inputs.q_nope.shape[1] == layer_config.num_heads, "q does not match the layer_config"
        kv, rank, rope = layer_inputs.kv_cache, layer_config.kv_lora_rank, layer_config.qk_rope_head_dim
        flash_attn_with_kvcache(
            q=layer_inputs.q_rope,
            k_cache=kv[:, :, rank:].view(-1, 1, 1, rope),
            v_cache=kv[:, :, :rank].view(-1, 1, 1, rank),
            qv=layer_inputs.q_nope,
            page_table=layer_inputs.topk_mem_indices,
            cache_seqlens=step_args.cache_seqlens,
            cu_seqlens_q=step_args.cu_seqlens_q,
            cu_seqlens_k_new=step_args.cu_seqlens_k,
            max_seqlen_q=step_args.max_seqlen_q,
            softmax_scale=layer_config.softmax_scale,
            causal=True,
            out=out,
        )
        return out


class FA3SparseMLADecodeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "fa3_sparse_mla_decode"

    @staticmethod
    def get_step_builder_cls() -> type[FA3SparseMLADecodeStepBuilder]:
        return FA3SparseMLADecodeStepBuilder

    @staticmethod
    def get_impl_cls() -> type[FA3SparseMLADecodeImpl]:
        return FA3SparseMLADecodeImpl

"""FA3 sparse MLA: the top-k cache slots serve as a one-token-per-page page table.

Decode covers bf16 caches, with or without rotary channels; prefill covers GLM-5's packed FP8 cache,
which it dequantizes chunk by chunk.
"""

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBatch,
    AttentionImpl,
    AttentionStepBuilder,
    StepBuilderContext,
    TensorAllocator,
)
from .. import keep_for_cuda_graph
from . import SparseMLADecodeLayerInputs, SparseMLALayerConfig, SparseMLAPrefillLayerInputs


@dataclass(frozen=True)
class FA3SparseMLADecodeStepArgs:
    """cache_seqlens is the number of selected slots each request reads; cu_seqlens_k is its int32
    prefix sum and cu_seqlens_q the int32 query offsets. causal is the kernel's mask mode, which
    does not change one-row decode."""

    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    causal: bool = True


class FA3SparseMLADecodeStepBuilder(AttentionStepBuilder[FA3SparseMLADecodeStepArgs]):
    def __init__(self, *, topk: int, kpool: int = 1, causal: bool = True) -> None:
        """topk is the indexer's selection width. An indexer that scores pools of kpool keys selects
        whole pools within the top-k width plus the newest partial pool."""
        self.topk = topk
        self.kpool = kpool
        self.causal = causal
        self._cuda_graph_tensors: dict[tuple, torch.Tensor] = {}

    def build(self, batch: AttentionBatch) -> FA3SparseMLADecodeStepArgs:
        """A CUDA graph step keeps the length tensors per (microbatch, batch size), rewritten in place."""
        device, lengths, kpool = batch.b_seq_len.device, batch.b_seq_len, self.kpool
        topk = torch.full((batch.batch_size,), self.topk, dtype=lengths.dtype, device=device)
        if kpool == 1:
            cache_seqlens = torch.minimum(topk, lengths)
        else:
            cache_seqlens = torch.minimum(lengths // kpool * kpool, topk) + lengths % kpool
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
            causal=self.causal,
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
        if rope == 0:
            # NoPE: only_qv attends with the latent query and cache alone.
            result = flash_attn_with_kvcache(
                q=None,
                qv=layer_inputs.q_nope,
                k_cache=None,
                v_cache=kv.view(-1, 1, 1, rank),
                page_table=layer_inputs.topk_mem_indices,
                cache_seqlens=step_args.cache_seqlens,
                cu_seqlens_q=step_args.cu_seqlens_q,
                cu_seqlens_k_new=step_args.cu_seqlens_k,
                max_seqlen_q=step_args.max_seqlen_q,
                softmax_scale=layer_config.softmax_scale,
                causal=step_args.causal,
                only_qv=True,
            )
            out.copy_(result)
            return out
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
            causal=step_args.causal,
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

    @classmethod
    def create_step_builder(
        cls, layer_config: SparseMLALayerConfig, ctx: StepBuilderContext
    ) -> FA3SparseMLADecodeStepBuilder:
        if layer_config.topk is None:
            raise ValueError("FA3 sparse MLA decode needs the layer config's topk")
        return FA3SparseMLADecodeStepBuilder(**cls.step_builder_key(layer_config)._asdict())

    @classmethod
    def step_builder_key(cls, layer_config: SparseMLALayerConfig) -> "_FA3DecodeKey":
        return _FA3DecodeKey(layer_config.topk, layer_config.kpool, layer_config.causal)


class _FA3DecodeKey(NamedTuple):
    topk: Optional[int]
    kpool: int
    causal: bool


@dataclass(frozen=True)
class FA3SparseMLAPrefillStepArgs:
    """Request tensors for gathering each chunk of the step's keys, plus caller-owned scratch.

    workspace holds "gathered_kv" [chunk_tokens, 1, kv_lora_rank] bf16, whose length sets the chunk
    size, "chunk_topk_indices" [query_tokens, topk_capacity], "chunk_cache_seqlens" [query_tokens]
    and "cu_seqlens_q" (0, 1, 2, ...).
    """

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    cu_q_seq_len: torch.Tensor
    cu_kv_seq_len: torch.Tensor
    total_token_num: int
    workspace: dict


class FA3SparseMLAPrefillStepBuilder(AttentionStepBuilder[FA3SparseMLAPrefillStepArgs]):
    def __init__(self, *, workspace: Callable[[int], dict]) -> None:
        """workspace(microbatch_index) returns that microbatch's caller-owned scratch."""
        self.workspace = workspace

    def build(self, batch: AttentionBatch) -> FA3SparseMLAPrefillStepArgs:
        return FA3SparseMLAPrefillStepArgs(
            req_to_token_indexs=batch.req_to_token_indexs,
            b_req_idx=batch.b_req_idx,
            cu_q_seq_len=batch.cu_q_seq_len,
            cu_kv_seq_len=batch.cu_kv_seq_len,
            total_token_num=batch.total_token_num,
            workspace=self.workspace(batch.microbatch_index),
        )


class FA3SparseMLAPrefillImpl(
    AttentionImpl[SparseMLALayerConfig, FA3SparseMLAPrefillStepArgs, SparseMLAPrefillLayerInputs]
):
    def __init__(self, layer_config: SparseMLALayerConfig) -> None:
        super().__init__(layer_config)
        if layer_config.kv_cache_format != "fp8_group64_e8m0" or layer_config.qk_rope_head_dim != 0:
            raise NotImplementedError("FA3 sparse MLA prefill supports only the NoPE fp8_group64_e8m0 cache")

    @torch.no_grad()
    def forward(
        self,
        step_args: FA3SparseMLAPrefillStepArgs,
        layer_inputs: SparseMLAPrefillLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [query_tokens, num_heads, kv_lora_rank].

        The step's keys (cached history dequantized next to current_kv) are gathered in chunks into
        the workspace; each chunk's partial attention merges into the output in FP32.
        """
        import triton
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        from lightllm.common.attention.kernels.sparse_mla.fp8_group64_prefill import (
            _gather_prefill_kv_kernel,
            _compact_topk_indices_kernel,
            _merge_att_output_kernel,
        )

        q, packed_kv = layer_inputs.q, layer_inputs.kv_cache
        prefill_cache_kv, topk_indices = layer_inputs.current_kv, layer_inputs.topk_indices
        assert prefill_cache_kv is not None and topk_indices is not None and layer_inputs.q_kv_start_loc is not None
        q_token_num, q_head_num, _ = q.shape
        buffers = step_args.workspace
        topk_size = topk_indices.shape[1]
        assert q_token_num <= buffers["chunk_cache_seqlens"].numel()
        assert topk_size <= buffers["chunk_topk_indices"].shape[1]
        gathered_kv = buffers["gathered_kv"]
        chunk_topk_indices = buffers["chunk_topk_indices"].view(-1)[: q_token_num * topk_size]
        chunk_topk_indices = chunk_topk_indices.view(q_token_num, topk_size)
        chunk_cache_seqlens = buffers["chunk_cache_seqlens"][:q_token_num]
        cu_seqlens_q = buffers["cu_seqlens_q"][: q_token_num + 1]
        req_to_token_indexs, b_req_idx = step_args.req_to_token_indexs, step_args.b_req_idx
        total_token_num = step_args.total_token_num
        gather_chunk_size = min(gathered_kv.shape[0], total_token_num)
        has_history = total_token_num > q_token_num
        if not has_history and prefill_cache_kv.stride(-1) != 1:
            prefill_cache_kv = prefill_cache_kv.contiguous()
        output = softmax_lse = None
        for chunk_start in range(0, total_token_num, gather_chunk_size):
            chunk_token_num = min(gather_chunk_size, total_token_num - chunk_start)
            if has_history:
                _gather_prefill_kv_kernel[(chunk_token_num // 2 + b_req_idx.numel(),)](
                    packed_kv,
                    prefill_cache_kv,
                    gathered_kv,
                    req_to_token_indexs,
                    b_req_idx,
                    step_args.cu_kv_seq_len,
                    step_args.cu_q_seq_len,
                    chunk_start,
                    chunk_token_num,
                    packed_kv.stride(0),
                    prefill_cache_kv.stride(0),
                    prefill_cache_kv.stride(2),
                    req_to_token_indexs_stride_b=req_to_token_indexs.stride(0),
                    req_to_token_indexs_stride_s=req_to_token_indexs.stride(1),
                    batch_size=b_req_idx.numel(),
                    BLOCK_ROWS=2,
                    num_warps=2,
                )
                chunk_kv = gathered_kv[:chunk_token_num].view(-1, 1, 1, 512)
            else:
                # The step's own keys already form the complete request layout.
                chunk_kv = prefill_cache_kv[chunk_start : chunk_start + chunk_token_num].view(-1, 1, 1, 512)
            _compact_topk_indices_kernel[(q_token_num,)](
                topk_indices,
                layer_inputs.q_kv_start_loc,
                chunk_topk_indices,
                chunk_cache_seqlens,
                chunk_start,
                chunk_start + chunk_token_num,
                topk_indices.stride(0),
                topk_indices.stride(1),
                TOPK=topk_size,
                BLOCK=triton.next_power_of_2(topk_size),
            )
            chunk_output, chunk_softmax_lse, *_ = flash_attn_with_kvcache(
                # only_qv ignores Q/K; these views avoid allocating placeholder tensors.
                q=q[..., :64],
                k_cache=chunk_kv[..., :64],
                qv=q,
                v_cache=chunk_kv,
                page_table=chunk_topk_indices,
                cache_seqlens=chunk_cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                softmax_scale=self.layer_config.softmax_scale,
                causal=False,
                only_qv=True,
                num_splits=1,
                return_softmax_lse=True,
            )
            if chunk_start == 0:
                output, softmax_lse = chunk_output, chunk_softmax_lse
            else:
                # Keep the online reduction in FP32 across all remaining chunks.
                output = output.float()
                _merge_att_output_kernel[(q_token_num, triton.cdiv(q_head_num, 4))](
                    output,
                    softmax_lse,
                    chunk_output,
                    chunk_softmax_lse,
                    chunk_softmax_lse.stride(1),
                    chunk_softmax_lse.stride(0),
                    q_head_num=q_head_num,
                    q_token_num=q_token_num,
                    BLOCK_HEADS=4,
                )
            del chunk_output, chunk_softmax_lse
        out.copy_(output)
        return out


class FA3SparseMLAPrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "fa3_sparse_mla_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[FA3SparseMLAPrefillStepBuilder]:
        return FA3SparseMLAPrefillStepBuilder

    @staticmethod
    def get_impl_cls() -> type[FA3SparseMLAPrefillImpl]:
        return FA3SparseMLAPrefillImpl

    @classmethod
    def create_step_builder(
        cls, layer_config: SparseMLALayerConfig, ctx: StepBuilderContext
    ) -> FA3SparseMLAPrefillStepBuilder:
        # The model reserves the scratch before it sizes the KV cache.
        if ctx.prefill_workspace is None:
            raise ValueError("FA3 sparse MLA prefill needs the model's prefill workspace (StepBuilderContext)")
        return FA3SparseMLAPrefillStepBuilder(workspace=ctx.prefill_workspace)

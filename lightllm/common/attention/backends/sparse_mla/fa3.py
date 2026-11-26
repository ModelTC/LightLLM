"""FA3 sparse MLA: the top-k cache slots serve as a one-token-per-page page table.

Decode covers bf16 caches, with or without rotary channels; prefill covers GLM-5's packed FP8 cache,
which it dequantizes chunk by chunk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from . import SparseMLADecodeLayerInputs, SparseMLALayerConfig, SparseMLAPrefillLayerInputs

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


@dataclass(frozen=True)
class FA3SparseMLAPrefillAttState:
    """The step's requests: request i's keys start at cu_kv_seq_len[i] of the step's ragged key layout and
    its queries at cu_q_seq_len[i]; total_token_num is the number of keys over all requests."""

    req_to_token_indexs: torch.Tensor
    b_req_idx: torch.Tensor
    total_token_num: int
    cu_kv_seq_len: torch.Tensor
    cu_q_seq_len: torch.Tensor


@dataclass(frozen=True)
class FA3SparseMLADecodeAttState:
    """cache_seqlens is the number of selected slots each request reads; cu_seqlens_k is its int32
    prefix sum and cu_seqlens_q the int32 query offsets."""

    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int


class FA3SparseMLAPrefillBackend(
    AttentionBackend[SparseMLALayerConfig, FA3SparseMLAPrefillAttState, SparseMLAPrefillLayerInputs]
):
    name = "fa3_sparse_mla_prefill"
    phase = "prefill"

    @classmethod
    def _check_supported(cls, layer_config: SparseMLALayerConfig) -> None:
        if layer_config.kv_cache_format != "fp8_group64_e8m0" or layer_config.qk_rope_head_dim != 0:
            raise NotImplementedError("FA3 sparse MLA prefill supports only the NoPE fp8_group64_e8m0 cache")

    def build(self, infer_state: InferStateInfo) -> FA3SparseMLAPrefillAttState:
        return FA3SparseMLAPrefillAttState(
            req_to_token_indexs=infer_state.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            total_token_num=infer_state.total_token_num,
            cu_kv_seq_len=infer_state.b1_cu_kv_seq_len,
            cu_q_seq_len=infer_state.b1_cu_q_seq_len,
        )

    @torch.no_grad()
    def forward(
        self,
        att_state: FA3SparseMLAPrefillAttState,
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
        current_kv, topk_indices = layer_inputs.current_kv, layer_inputs.topk_indices
        assert current_kv is not None and topk_indices is not None and layer_inputs.q_kv_start_loc is not None
        buffers = layer_inputs.fp8_workspace
        if buffers is None:
            raise ValueError("FA3 sparse MLA prefill requires the reserved FP8 workspace")
        q_token_num, q_head_num, _ = q.shape
        topk_size = topk_indices.shape[1]
        assert q_token_num <= buffers["chunk_cache_seqlens"].numel()
        assert topk_size <= buffers["chunk_topk_indices"].shape[1]

        # The workspace, viewed at this step's sizes.
        gathered_kv = buffers["gathered_kv"]
        chunk_topk_indices = buffers["chunk_topk_indices"].view(-1)[: q_token_num * topk_size]
        chunk_topk_indices = chunk_topk_indices.view(q_token_num, topk_size)
        chunk_cache_seqlens = buffers["chunk_cache_seqlens"][:q_token_num]
        cu_seqlens_q = buffers["cu_seqlens_q"][: q_token_num + 1]

        req_to_token_indexs, b_req_idx = att_state.req_to_token_indexs, att_state.b_req_idx
        total_token_num = att_state.total_token_num
        has_history = total_token_num > q_token_num
        if not has_history and current_kv.stride(-1) != 1:
            current_kv = current_kv.contiguous()
        gather_chunk_size = min(gathered_kv.shape[0], total_token_num)
        # One chunk writes out directly; several merge their outputs in FP32 first.
        single_chunk = gather_chunk_size >= total_token_num

        output = softmax_lse = None
        for chunk_start in range(0, total_token_num, gather_chunk_size):
            chunk_token_num = min(gather_chunk_size, total_token_num - chunk_start)
            if has_history:
                _gather_prefill_kv_kernel[(chunk_token_num // 2 + b_req_idx.numel(),)](
                    packed_kv,
                    current_kv,
                    gathered_kv,
                    req_to_token_indexs,
                    b_req_idx,
                    att_state.cu_kv_seq_len,
                    att_state.cu_q_seq_len,
                    chunk_start,
                    chunk_token_num,
                    packed_kv.stride(0),
                    current_kv.stride(0),
                    current_kv.stride(2),
                    req_to_token_indexs_stride_b=req_to_token_indexs.stride(0),
                    req_to_token_indexs_stride_s=req_to_token_indexs.stride(1),
                    batch_size=b_req_idx.numel(),
                    BLOCK_ROWS=2,
                    num_warps=2,
                )
                chunk_kv = gathered_kv[:chunk_token_num].view(-1, 1, 1, 512)
            else:
                # The step's own keys already form the complete request layout.
                chunk_kv = current_kv[chunk_start : chunk_start + chunk_token_num].view(-1, 1, 1, 512)
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
                out=out if single_chunk else None,
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
        if not single_chunk:
            out.copy_(output)
        return out


class FA3SparseMLADecodeBackend(
    AttentionBackend[SparseMLALayerConfig, FA3SparseMLADecodeAttState, SparseMLADecodeLayerInputs]
):
    """The layer config's causal is the kernel's mask mode, which does not change one-row decode."""

    name = "fa3_sparse_mla_decode"
    phase = "decode"

    @classmethod
    def _check_supported(cls, layer_config: SparseMLALayerConfig) -> None:
        if layer_config.kv_cache_format != "bf16":
            raise NotImplementedError("FA3 sparse MLA decode supports only the bf16 cache")
        if layer_config.topk is None:
            raise NotImplementedError("FA3 sparse MLA decode needs the layer config's topk")

    @classmethod
    def group_key(cls, layer_config: SparseMLALayerConfig):
        return layer_config.topk, layer_config.kpool, layer_config.causal

    def build(self, infer_state: InferStateInfo) -> FA3SparseMLADecodeAttState:
        """A CUDA graph step keeps the length tensors per (microbatch, batch size), rewritten in place.

        An indexer that scores pools of kpool keys selects whole pools within the top-k width plus the
        newest partial pool.
        """
        lengths, topk, kpool = infer_state.b_seq_len, self.layer_config.topk, self.layer_config.kpool
        topk = torch.full((infer_state.batch_size,), topk, dtype=lengths.dtype, device=lengths.device)
        if kpool == 1:
            cache_seqlens = torch.minimum(topk, lengths)
        else:
            cache_seqlens = torch.minimum(lengths // kpool * kpool, topk) + lengths % kpool
        padded = torch.zeros(infer_state.batch_size + 1, dtype=torch.int32, device=lengths.device)
        padded[1:].copy_(cache_seqlens, non_blocking=True)
        cu_seqlens_k = padded.cumsum(dim=0, dtype=torch.int32)
        return FA3SparseMLADecodeAttState(
            cache_seqlens=self.keep_for_graph(infer_state, "cache_seqlens", cache_seqlens),
            cu_seqlens_q=infer_state.b1_cu_q_seq_len,
            cu_seqlens_k=self.keep_for_graph(infer_state, "cu_seqlens_k", cu_seqlens_k),
            max_seqlen_q=infer_state.max_q_seq_len,
        )

    def forward(
        self,
        att_state: FA3SparseMLADecodeAttState,
        layer_inputs: SparseMLADecodeLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out with q_nope's shape."""
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        config, q_nope = self.layer_config, layer_inputs.q_nope
        assert q_nope.shape[1] == config.num_heads, "q does not match the layer_config"
        kv, rank, rope = layer_inputs.kv_cache, config.kv_lora_rank, config.qk_rope_head_dim
        selected = dict(
            page_table=layer_inputs.topk_mem_indices,
            cache_seqlens=att_state.cache_seqlens,
            cu_seqlens_q=att_state.cu_seqlens_q,
            cu_seqlens_k_new=att_state.cu_seqlens_k,
            max_seqlen_q=att_state.max_seqlen_q,
            softmax_scale=config.softmax_scale,
            causal=config.causal,
            out=out,
        )
        if rope == 0:
            # NoPE: only_qv attends with the latent query and cache alone.
            flash_attn_with_kvcache(
                q=None, qv=q_nope, k_cache=None, v_cache=kv.view(-1, 1, 1, rank), only_qv=True, **selected
            )
        else:
            flash_attn_with_kvcache(
                q=layer_inputs.q_rope,
                k_cache=kv[:, :, rank:].view(-1, 1, 1, rope),
                v_cache=kv[:, :, :rank].view(-1, 1, 1, rank),
                qv=q_nope,
                **selected,
            )
        return out

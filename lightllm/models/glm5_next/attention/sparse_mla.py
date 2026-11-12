import dataclasses
import inspect

import torch

from lightllm.common.attention.backends.sparse_mla.flashmla.backend import (
    NsaFlashMlaSparseAttBackend,
    NsaFlashMlaSparseDecodeContext,
    NsaFlashMlaSparsePrefillContext,
)


@dataclasses.dataclass(slots=True)
class Glm5NextSparsePrefillContext(NsaFlashMlaSparsePrefillContext):
    pass


@dataclasses.dataclass(slots=True)
class Glm5NextSparseDecodeContext(NsaFlashMlaSparseDecodeContext):
    pass


class Glm5NextSparseAttBackend(NsaFlashMlaSparseAttBackend):
    provider = "flashmla_sparse"
    cache_formats = frozenset({"None"})

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return Glm5NextSparsePrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return Glm5NextSparseDecodeContext(capacity, capture, workspace)

    def _prefill_nsa_prefill_att(self, layer, context, q, kv, inputs):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        tokens, heads, dim = q.shape
        padded_heads = (heads + 63) // 64 * 64
        padded_q = q
        if padded_heads != heads:
            padded_q = q.new_zeros((tokens, padded_heads, dim))
            padded_q[:, :heads] = q
        out, _, _ = flash_mla_sparse_fwd(
            q=padded_q,
            kv=kv,
            indices=inputs.topk_mem_indices.unsqueeze(1),
            sm_scale=layer.params.softmax_scale,
            d_v=layer.params.kv_lora_rank,
        )
        return out[:, :heads]

    def _update_decode(self, context):
        super()._update_decode(context)
        pool = self.network_config["index_kpool"]
        topk = self.network_config["index_topk"]
        context.nsa_cache_seqlens = (
            torch.minimum(context.lengths // pool * pool, torch.full_like(context.lengths, topk))
            + context.lengths % pool
        )
        context.nsa_cu_seqlens_k_new = torch.nn.functional.pad(
            context.nsa_cache_seqlens.cumsum(0, dtype=torch.int32), (1, 0)
        )

    def _decode_nsa_decode_att(self, layer, context, q, kv, inputs):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        q_nope, _ = q
        kv_nope = kv.view(-1, 1, 1, layer.params.kv_lora_rank)
        common_kwargs = dict(
            page_table=inputs.topk_mem_indices,
            cache_seqlens=context.nsa_cache_seqlens,
            cu_seqlens_q=context.batch.b1_cu_q_seq_len,
            cu_seqlens_k_new=context.nsa_cu_seqlens_k_new,
            max_seqlen_q=context.batch.max_q_seq_len,
            softmax_scale=layer.params.softmax_scale,
            causal=False,
        )
        if "only_qv" in inspect.signature(flash_attn_with_kvcache).parameters:
            return flash_attn_with_kvcache(
                q=None, qv=q_nope, k_cache=None, v_cache=kv_nope, only_qv=True, **common_kwargs
            )
        if q_nope.shape[-1] > 256:
            page_indices = inputs.topk_mem_indices.clamp_min(0)
            key = kv_nope[page_indices, 0, 0]
            score = torch.einsum("bhd,bkd->bhk", q_nope.float(), key.float())
            score = score * layer.params.softmax_scale
            valid = (
                torch.arange(page_indices.shape[1], device=page_indices.device)[None, :]
                < context.nsa_cache_seqlens[:, None]
            )
            score = score.masked_fill(~valid[:, None, :], float("-inf"))
            weights = score.softmax(dim=-1)
            weights = torch.where(valid[:, None, :], weights, torch.zeros_like(weights))
            return torch.einsum("bhk,bkd->bhd", weights, key.float()).to(q_nope.dtype)
        return flash_attn_with_kvcache(q=q_nope, k_cache=kv_nope, v_cache=kv_nope, **common_kwargs)

import dataclasses

import torch

from .flashmla_sparse import (
    NsaFlashMlaSparseAttBackend,
    NsaFlashMlaSparsePrefillAttState,
    NsaFlashMlaSparseDecodeAttState,
)


class Glm5NextSparseAttBackend(NsaFlashMlaSparseAttBackend):
    def create_att_prefill_state(self, infer_state):
        return Glm5NextSparsePrefillState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        return Glm5NextSparseDecodeState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class Glm5NextSparsePrefillState(NsaFlashMlaSparsePrefillAttState):
    def _nsa_prefill_att(self, q, kv, att_control):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        tokens, heads, dim = q.shape
        # FlashMLA accepts native 512-wide NoPE Q/K; head counts still use 64-head tiles.
        padded_heads = ((heads + 63) // 64) * 64
        padded_q = q
        if padded_heads != heads:
            padded_q = q.new_zeros((tokens, padded_heads, dim))
            padded_q[:, :heads] = q
        params = att_control.nsa_prefill_dict
        out, _, _ = flash_mla_sparse_fwd(
            q=padded_q,
            kv=kv,
            indices=params["topk_mem_indices"].unsqueeze(1),
            sm_scale=params["softmax_scale"],
            d_v=512,
        )
        return out[:, :heads]


@dataclasses.dataclass
class Glm5NextSparseDecodeState(NsaFlashMlaSparseDecodeAttState):
    def init_state(self):
        super().init_state()
        pool = self.backend.model.config["index_kpool"]
        topk = self.backend.model.config["index_topk"]
        self.nsa_cache_seqlens = (
            torch.minimum(self.lengths // pool * pool, torch.full_like(self.lengths, topk)) + self.lengths % pool
        )
        self.nsa_cu_seqlens_k_new = torch.nn.functional.pad(self.nsa_cache_seqlens.cumsum(0, dtype=torch.int32), (1, 0))

    def _nsa_decode_att(self, q, kv, att_control):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        q_nope, _ = q
        kv_nope = kv.view(-1, 1, 1, 512)
        params = att_control.nsa_decode_dict
        # only_qv skips QK entirely. Reuse views for the API's required Q/K tensors
        # so the wrapper does not allocate a dummy 64-wide query or KV cache.
        return flash_attn_with_kvcache(
            q=q_nope[..., :64],
            qv=q_nope,
            k_cache=kv_nope[..., :64],
            v_cache=kv_nope,
            page_table=params["topk_mem_indices"],
            cache_seqlens=self.nsa_cache_seqlens,
            cu_seqlens_q=self.infer_state.b1_cu_q_seq_len,
            cu_seqlens_k_new=self.nsa_cu_seqlens_k_new,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            softmax_scale=params["softmax_scale"],
            causal=False,
            only_qv=True,
        )

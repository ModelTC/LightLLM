import dataclasses
import weakref

import torch

from ..base_att import AttControl, BaseAttBackend, BasePrefillAttState, BaseDecodeAttState


class Glm5NextSparseAttBackend(BaseAttBackend):
    def create_att_prefill_state(self, infer_state):
        return Glm5NextSparsePrefillState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state):
        return Glm5NextSparseDecodeState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class Glm5NextSparsePrefillState(BasePrefillAttState):
    lengths: torch.Tensor = None
    query_req: torch.Tensor = None
    ks: torch.Tensor = None

    def init_state(self):
        from lightllm.models.glm5_next.triton_kernel.kpool import prefill_nsa_params

        state = self.infer_state
        self.lengths, self.query_req, self.ks = prefill_nsa_params(
            state.position_ids, state.b_req_idx, state.b1_cu_q_seq_len, state.b1_cu_kv_seq_len
        )

    def prefill_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        assert att_control.nsa_prefill and att_control.nsa_prefill_dict is not None
        return self._nsa_prefill_att(q=q, kv=k, att_control=att_control)

    def _nsa_prefill_att(self, q, kv, att_control):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        tokens, heads, dim = q.shape
        # FlashMLA uses 64-head tiles.
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
            d_v=params["kv_lora_rank"],
        )
        return out[:, :heads]


@dataclasses.dataclass
class Glm5NextSparseDecodeState(BaseDecodeAttState):
    lengths: torch.Tensor = None
    query_req: torch.Tensor = None
    nsa_cache_seqlens: torch.Tensor = None
    nsa_cu_seqlens_k_new: torch.Tensor = None

    def init_state(self):
        self.lengths = self.infer_state.b_seq_len
        self.query_req = self.infer_state.b_req_idx
        pool = self.backend.model.config["index_kpool"]
        topk = self.backend.model.config["index_topk"]
        self.nsa_cache_seqlens = (
            torch.minimum(self.lengths // pool * pool, torch.full_like(self.lengths, topk)) + self.lengths % pool
        )
        self.nsa_cu_seqlens_k_new = torch.nn.functional.pad(self.nsa_cache_seqlens.cumsum(0, dtype=torch.int32), (1, 0))

    def decode_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        assert att_control.nsa_decode and att_control.nsa_decode_dict is not None
        return self._nsa_decode_att(q, k, att_control)

    def _nsa_decode_att(self, q, kv, att_control):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        params = att_control.nsa_decode_dict
        q_nope, _ = q
        kv_nope = kv.view(-1, 1, 1, params["kv_lora_rank"])
        # NoPE has no RoPE Q/K tensors; only_qv uses q_nope and kv_nope directly.
        return flash_attn_with_kvcache(
            q=None,
            qv=q_nope,
            k_cache=None,
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

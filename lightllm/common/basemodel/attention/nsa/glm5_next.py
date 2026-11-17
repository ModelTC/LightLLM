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
    q_kv_seq_len: torch.Tensor = None
    q_req_idx: torch.Tensor = None
    q_kv_start_loc: torch.Tensor = None

    def init_state(self):
        from lightllm.models.glm5_next.triton_kernel.kpool import prefill_nsa_params

        infer_state = self.infer_state
        self.q_kv_seq_len, self.q_req_idx, self.q_kv_start_loc = prefill_nsa_params(
            infer_state.position_ids, infer_state.b_req_idx, infer_state.b1_cu_q_seq_len, infer_state.b1_cu_kv_seq_len
        )

    def prefill_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        assert att_control.nsa_prefill and att_control.nsa_prefill_dict is not None
        return self._nsa_prefill_att(q, k, att_control)

    def _nsa_prefill_att(self, q, kv, att_control):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        q_token_num, q_head_num, head_dim = q.shape
        # FlashMLA uses 64-head tiles.
        padded_q_head_num = ((q_head_num + 63) // 64) * 64
        padded_q = q
        if padded_q_head_num != q_head_num:
            padded_q = q.new_zeros((q_token_num, padded_q_head_num, head_dim))
            padded_q[:, :q_head_num] = q
        nsa_prefill_dict = att_control.nsa_prefill_dict
        output, _, _ = flash_mla_sparse_fwd(
            q=padded_q,
            kv=kv,
            indices=nsa_prefill_dict["topk_mem_indices"].unsqueeze(1),
            sm_scale=nsa_prefill_dict["softmax_scale"],
            d_v=nsa_prefill_dict["kv_lora_rank"],
        )
        return output[:, :q_head_num]


@dataclasses.dataclass
class Glm5NextSparseDecodeState(BaseDecodeAttState):
    q_kv_seq_len: torch.Tensor = None
    q_req_idx: torch.Tensor = None
    nsa_cache_seqlens: torch.Tensor = None
    nsa_cu_seqlens_k_new: torch.Tensor = None

    def init_state(self):
        self.q_kv_seq_len = self.infer_state.b_seq_len
        self.q_req_idx = self.infer_state.b_req_idx
        index_kpool = self.backend.model.config["index_kpool"]
        index_topk = self.backend.model.config["index_topk"]
        self.nsa_cache_seqlens = (
            torch.minimum(
                self.q_kv_seq_len // index_kpool * index_kpool, torch.full_like(self.q_kv_seq_len, index_topk)
            )
            + self.q_kv_seq_len % index_kpool
        )
        self.nsa_cu_seqlens_k_new = torch.nn.functional.pad(self.nsa_cache_seqlens.cumsum(0, dtype=torch.int32), (1, 0))

    def decode_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        assert att_control.nsa_decode and att_control.nsa_decode_dict is not None
        return self._nsa_decode_att(q, k, att_control)

    def _nsa_decode_att(self, q, kv, att_control):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        nsa_decode_dict = att_control.nsa_decode_dict
        q_nope, _ = q
        kv_nope = kv.view(-1, 1, 1, nsa_decode_dict["kv_lora_rank"])
        # NoPE has no RoPE Q/K tensors; only_qv uses q_nope and kv_nope directly.
        return flash_attn_with_kvcache(
            q=None,
            qv=q_nope,
            k_cache=None,
            v_cache=kv_nope,
            page_table=nsa_decode_dict["topk_mem_indices"],
            cache_seqlens=self.nsa_cache_seqlens,
            cu_seqlens_q=self.infer_state.b1_cu_q_seq_len,
            cu_seqlens_k_new=self.nsa_cu_seqlens_k_new,
            max_seqlen_q=self.infer_state.max_q_seq_len,
            softmax_scale=nsa_decode_dict["softmax_scale"],
            causal=False,
            only_qv=True,
        )

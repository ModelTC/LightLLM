import dataclasses

import torch

from .fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend, NsaFlashMlaFp8SparseDecodeAttState
from .glm5_next import Glm5NextSparsePrefillState


class Fp8Glm5NextSparseAttBackend(NsaFlashMlaFp8SparseAttBackend):
    def create_att_prefill_state(self, infer_state):
        return Fp8Glm5NextSparsePrefillState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state):
        return Fp8Glm5NextSparseDecodeState(backend=self, infer_state=infer_state)


@dataclasses.dataclass
class Fp8Glm5NextSparsePrefillState(Glm5NextSparsePrefillState):
    def _nsa_prefill_att(self, q, kv, att_control):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            context_attention_fwd_fp8,
        )

        params = att_control.nsa_prefill_dict
        return context_attention_fwd_fp8(
            q,
            kv,
            self.ragged_mem_index,
            self.infer_state.mem_index,
            params["prefill_cache_kv"],
            self.ks,
            params["topk_indices"],
            params["softmax_scale"],
        )


@dataclasses.dataclass
class Fp8Glm5NextSparseDecodeState(NsaFlashMlaFp8SparseDecodeAttState):
    def _nsa_decode_att(self, q, packed_kv, att_control):
        import flash_mla

        params = att_control.nsa_decode_dict
        topk_mem_indices = params["topk_mem_indices"]
        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        assert topk_mem_indices.shape[1] == 1, "FlashMLA sparse decode path currently expects seq_len_q == 1"

        q_nope, _ = q
        num_tokens, num_heads, _ = q_nope.shape
        # FlashMLA uses 64-head tiles; NoPE keeps its original 512 dimensions.
        padded_num_heads = ((num_heads + 63) // 64) * 64
        padded_q = q_nope.unsqueeze(1).contiguous()
        if padded_num_heads != num_heads:
            padded_q = q_nope.new_zeros((num_tokens, 1, padded_num_heads, 512))
            padded_q[:, 0, :num_heads] = q_nope
        kv_cache = torch.as_strided(
            packed_kv,
            size=(packed_kv.shape[0], 1, 1, packed_kv.shape[-1]),
            stride=(packed_kv.stride(0), packed_kv.shape[-1], packed_kv.shape[-1], packed_kv.stride(-1)),
        )
        output, _ = flash_mla.flash_mla_with_kvcache(
            q=padded_q,
            k_cache=kv_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=params["kv_lora_rank"],
            tile_scheduler_metadata=self.flashmla_sched_meta,
            softmax_scale=params["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
        )
        return output[:, 0, :num_heads, :]

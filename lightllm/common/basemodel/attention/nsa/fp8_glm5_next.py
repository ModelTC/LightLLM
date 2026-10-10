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
        from lightllm.models.glm5_next.triton_kernel.prefill_gather_kv_flashmla_fp8 import (
            gather_prefill_kv_cache_triton,
        )

        params = att_control.nsa_prefill_dict
        prefill_kv = params["prefill_cache_kv"]
        if self.infer_state.max_cache_len > 0:
            # Known ragged indices avoid host synchronization during graph capture.
            prefill_kv = gather_prefill_kv_cache_triton(
                kv, self.ragged_mem_index, self.infer_state.mem_index, prefill_kv
            )
        # Indexer top-k entries are relative to each request, not the batch.
        topk_indices = params["topk_indices"]
        indices = torch.where(topk_indices >= 0, topk_indices + self.ks[:, None], -1)
        prefill_control = dataclasses.replace(att_control, nsa_prefill_dict={**params, "topk_mem_indices": indices})
        return super()._nsa_prefill_att(q, prefill_kv, prefill_control)


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
        # NoPE uses V3.2's zero RoPE tail and FlashMLA's 64-head tiles.
        padded_num_heads = ((num_heads + 63) // 64) * 64
        q_all = q_nope.new_zeros((num_tokens, 1, padded_num_heads, 576))
        q_all[:, 0, :num_heads, :512] = q_nope
        kv = torch.as_strided(
            packed_kv,
            size=(packed_kv.shape[0], 1, 1, packed_kv.shape[-1]),
            stride=(packed_kv.stride(0), packed_kv.shape[-1], packed_kv.shape[-1], packed_kv.stride(-1)),
        )
        output, _ = flash_mla.flash_mla_with_kvcache(
            q=q_all,
            k_cache=kv,
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

import dataclasses
import weakref

import torch

from .glm5_next import Glm5NextSparsePrefillState, Glm5NextSparseDecodeState
from ..base_att import BaseAttBackend


class Fp8Glm5NextSparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        self._prefill_workspaces = [{"smallq": None, "generic": None} for _ in range(2)]

    def get_smallq_workspace(self, infer_state, q):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            allocate_direct_smallq_workspace,
            allocate_selected_prefill_workspace,
        )

        slot = self._prefill_workspaces[infer_state.microbatch_index]
        if slot["smallq"] is None:
            slot["smallq"] = {
                "direct": allocate_direct_smallq_workspace(q[:1]),
                "selected": allocate_selected_prefill_workspace(q, q_token_capacity=8),
            }
        if q.shape[0] == 1:
            return slot["smallq"]["direct"]
        q_token_num = q.shape[0]
        selected = slot["smallq"]["selected"]
        topk_size = selected["compact_topk_mem_indices"].shape[1]
        return {
            "compact_topk_mem_indices": selected["compact_topk_mem_indices"][:q_token_num],
            "cache_seqlens": selected["cache_seqlens"][:q_token_num],
            "page_table": selected["page_table"][:q_token_num],
            "cu_seqlens_q": selected["cu_seqlens_q"][: q_token_num + 1],
            "cu_seqlens_k_new": selected["cu_seqlens_k_new"][: q_token_num + 1],
            "selected_kv": selected["selected_kv"][: q_token_num * topk_size],
        }

    def get_generic_prefill_workspace(self, infer_state, q, topk_indices):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            allocate_generic_prefill_workspace,
        )

        slot = self._prefill_workspaces[infer_state.microbatch_index]
        buffers = slot["generic"]
        q_token_num, topk_size = topk_indices.shape
        if (
            buffers is None
            or buffers["chunk_topk_indices"].shape[0] < q_token_num
            or buffers["chunk_topk_indices"].shape[1] != topk_size
            or buffers["chunk_topk_indices"].dtype != topk_indices.dtype
        ):
            buffers = allocate_generic_prefill_workspace(
                q, topk_indices, q_token_capacity=max(self.model.batch_max_tokens or 0, q_token_num)
            )
            slot["generic"] = buffers
        return {
            "chunk_topk_indices": buffers["chunk_topk_indices"][:q_token_num],
            "chunk_cache_seqlens": buffers["chunk_cache_seqlens"][:q_token_num],
            "cu_seqlens_q": buffers["cu_seqlens_q"][: q_token_num + 1],
        }

    def create_att_prefill_state(self, infer_state):
        return Fp8Glm5NextSparsePrefillState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state):
        config = self.model.config
        # 2048 pooled entries plus at most three kpool remainder entries fit the 2112 rounded direct window.
        bound = 2051 if config["index_topk"] == 2048 and config["index_kpool"] == 4 else None
        return Fp8Glm5NextSparseDecodeState(
            backend=self, _infer_state_weak_ref=weakref.ref(infer_state), decode_topk_bound=bound
        )


@dataclasses.dataclass
class Fp8Glm5NextSparsePrefillState(Glm5NextSparsePrefillState):
    def _nsa_prefill_att(self, q, packed_kv, att_control):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            context_attention_fwd_fp8,
            can_use_direct_smallq,
            can_use_selected_prefill,
        )

        nsa_prefill_dict = att_control.nsa_prefill_dict
        infer_state = self.infer_state
        topk_mem_indices = nsa_prefill_dict["topk_mem_indices"]
        direct_workspace = selected_workspace = generic_workspace = None
        if q.shape[1] == 64:
            eligibility = (
                q,
                packed_kv,
                infer_state.mem_index,
                nsa_prefill_dict["prefill_cache_kv"],
                topk_mem_indices,
                infer_state.total_token_num,
            )
            if can_use_direct_smallq(*eligibility):
                direct_workspace = self.backend.get_smallq_workspace(infer_state, q)
            elif can_use_selected_prefill(*eligibility):
                selected_workspace = self.backend.get_smallq_workspace(infer_state, q)
            else:
                graph_guard = (
                    infer_state.is_cuda_graph_capture
                    or self.backend.model.prefill_graph is not None
                    or hasattr(infer_state, "prefill_cuda_graph_exe_list")
                )
                if not graph_guard and not torch.cuda.is_current_stream_capturing():
                    generic_workspace = self.backend.get_generic_prefill_workspace(
                        infer_state, q, nsa_prefill_dict["topk_indices"]
                    )
        return context_attention_fwd_fp8(
            q=q,
            packed_kv=packed_kv,
            mem_index=infer_state.mem_index,
            prefill_cache_kv=nsa_prefill_dict["prefill_cache_kv"],
            q_kv_start_loc=self.q_kv_start_loc,
            topk_indices=nsa_prefill_dict["topk_indices"],
            topk_mem_indices=topk_mem_indices,
            req_to_token_indexs=infer_state.req_manager.req_to_token_indexs,
            b_req_idx=infer_state.b_req_idx,
            b1_cu_kv_seq_len=infer_state.b1_cu_kv_seq_len,
            b1_cu_q_seq_len=infer_state.b1_cu_q_seq_len,
            total_token_num=infer_state.total_token_num,
            softmax_scale=nsa_prefill_dict["softmax_scale"],
            selected_workspace=selected_workspace,
            direct_workspace=direct_workspace,
            generic_workspace=generic_workspace,
        )


@dataclasses.dataclass
class Fp8Glm5NextSparseDecodeState(Glm5NextSparseDecodeState):
    decode_topk_bound: int = None
    flashmla_sched_meta: object = None

    def init_state(self):
        import flash_mla

        self.q_kv_seq_len = self.infer_state.b_seq_len
        self.q_req_idx = self.infer_state.b_req_idx
        self.flashmla_sched_meta, _ = flash_mla.get_mla_metadata()

    def _nsa_decode_att(self, q, packed_kv, att_control):
        import flash_mla

        from lightllm.models.glm5_next.triton_kernel import fp8_sparse_decode
        from lightllm.models.glm5_next.triton_kernel.pad_query_heads import pad_query_heads

        nsa_decode_dict = att_control.nsa_decode_dict
        topk_mem_indices = nsa_decode_dict["topk_mem_indices"]
        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        assert topk_mem_indices.shape[1] == 1, "FlashMLA sparse decode path currently expects seq_len_q == 1"

        q_nope, _ = q
        num_heads = q_nope.shape[1]
        if fp8_sparse_decode.can_use_fp8_sparse_direct_decode(
            q_nope,
            packed_kv,
            topk_mem_indices,
            nsa_decode_dict["kv_lora_rank"],
            self.decode_topk_bound,
            is_cuda_graph_capture=self.infer_state.is_cuda_graph_capture,
        ):
            return fp8_sparse_decode.fp8_sparse_direct_decode(
                q_nope, packed_kv, topk_mem_indices, nsa_decode_dict["softmax_scale"]
            )
        padded_q = pad_query_heads(q_nope, is_cuda_graph_capture=self.infer_state.is_cuda_graph_capture)
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
            head_dim_v=nsa_decode_dict["kv_lora_rank"],
            tile_scheduler_metadata=self.flashmla_sched_meta,
            softmax_scale=nsa_decode_dict["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
        )
        return output[:, 0, :num_heads, :]

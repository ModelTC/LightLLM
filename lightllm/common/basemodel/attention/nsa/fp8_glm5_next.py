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
                "selected": allocate_selected_prefill_workspace(q, batch_capacity=8),
            }
        if q.shape[0] == 1:
            return slot["smallq"]["direct"]
        batch = q.shape[0]
        selected = slot["smallq"]["selected"]
        slots = selected["compact"].shape[1]
        return {
            "compact": selected["compact"][:batch],
            "lengths": selected["lengths"][:batch],
            "page_table": selected["page_table"][:batch],
            "cu_q": selected["cu_q"][: batch + 1],
            "cu_k": selected["cu_k"][: batch + 1],
            "selected_kv": selected["selected_kv"][: batch * slots],
        }

    def get_generic_prefill_workspace(self, infer_state, q, topk_indices):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            allocate_generic_prefill_workspace,
        )

        slot = self._prefill_workspaces[infer_state.microbatch_index]
        buffers = slot["generic"]
        rows, columns = topk_indices.shape
        if (
            buffers is None
            or buffers["chunk_indices"].shape[0] < rows
            or buffers["chunk_indices"].shape[1] != columns
            or buffers["chunk_indices"].dtype != topk_indices.dtype
        ):
            buffers = allocate_generic_prefill_workspace(
                q, topk_indices, query_capacity=max(self.model.batch_max_tokens or 0, rows)
            )
            slot["generic"] = buffers
        return {
            "chunk_indices": buffers["chunk_indices"][:rows],
            "chunk_lengths": buffers["chunk_lengths"][:rows],
            "cu_seqlens_q": buffers["cu_seqlens_q"][: rows + 1],
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
    def _nsa_prefill_att(self, q, kv, att_control):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            context_attention_fwd_fp8,
            can_use_direct_smallq,
            can_use_selected_prefill,
        )

        params = att_control.nsa_prefill_dict
        state = self.infer_state
        topk_mem_indices = params["topk_mem_indices"]
        direct_workspace = selected_workspace = generic_workspace = None
        if q.shape[1] == 64:
            eligibility = (q, kv, state.mem_index, params["prefill_cache_kv"], topk_mem_indices, state.total_token_num)
            if can_use_direct_smallq(*eligibility):
                direct_workspace = self.backend.get_smallq_workspace(state, q)
            elif can_use_selected_prefill(*eligibility):
                selected_workspace = self.backend.get_smallq_workspace(state, q)
            else:
                graph_guard = (
                    state.is_cuda_graph_capture
                    or self.backend.model.prefill_graph is not None
                    or hasattr(state, "prefill_cuda_graph_exe_list")
                )
                if not graph_guard and not torch.cuda.is_current_stream_capturing():
                    generic_workspace = self.backend.get_generic_prefill_workspace(state, q, params["topk_indices"])
        return context_attention_fwd_fp8(
            q=q,
            packed_kv=kv,
            prefill_mem_index=state.mem_index,
            prefill_kv=params["prefill_cache_kv"],
            kv_starts=self.ks,
            topk_indices=params["topk_indices"],
            topk_mem_indices=topk_mem_indices,
            req_to_token_index=state.req_manager.req_to_token_indexs,
            b_req_idx=state.b_req_idx,
            cu_kv_seq_len=state.b1_cu_kv_seq_len,
            cu_q_seq_len=state.b1_cu_q_seq_len,
            total_kv_tokens=state.total_token_num,
            softmax_scale=params["softmax_scale"],
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

        self.lengths = self.infer_state.b_seq_len
        self.query_req = self.infer_state.b_req_idx
        self.flashmla_sched_meta, _ = flash_mla.get_mla_metadata()

    def _nsa_decode_att(self, q, packed_kv, att_control):
        import flash_mla

        from lightllm.models.glm5_next.triton_kernel import fp8_sparse_decode
        from lightllm.models.glm5_next.triton_kernel.pad_query_heads import pad_query_heads

        params = att_control.nsa_decode_dict
        topk_mem_indices = params["topk_mem_indices"]
        if topk_mem_indices.ndim == 2:
            topk_mem_indices = topk_mem_indices.unsqueeze(1)
        assert topk_mem_indices.shape[1] == 1, "FlashMLA sparse decode path currently expects seq_len_q == 1"

        q_nope, _ = q
        num_heads = q_nope.shape[1]
        if fp8_sparse_decode.can_use_fp8_sparse_direct_decode(
            q_nope,
            packed_kv,
            topk_mem_indices,
            params["kv_lora_rank"],
            self.decode_topk_bound,
            is_cuda_graph_capture=self.infer_state.is_cuda_graph_capture,
        ):
            return fp8_sparse_decode.fp8_sparse_direct_decode(
                q_nope, packed_kv, topk_mem_indices, params["softmax_scale"]
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
            head_dim_v=params["kv_lora_rank"],
            tile_scheduler_metadata=self.flashmla_sched_meta,
            softmax_scale=params["softmax_scale"],
            causal=False,
            is_fp8_kvcache=True,
            indices=topk_mem_indices,
        )
        return output[:, 0, :num_heads, :]

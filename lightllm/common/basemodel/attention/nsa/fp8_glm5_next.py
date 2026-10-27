import dataclasses
import weakref

import torch

from .fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend, NsaFlashMlaFp8SparseDecodeAttState
from .glm5_next import Glm5NextSparsePrefillState


class Fp8Glm5NextSparseAttBackend(NsaFlashMlaFp8SparseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        # Match the backend's existing microbatch-separated ragged buffers.
        self._prefill_workspaces = [{"smallq": None, "generic": None} for _ in self.ragged_mem_buffers]

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

    def get_generic_prefill_workspace(self, infer_state, q, packed_kv, topk_indices):
        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            allocate_generic_prefill_workspace,
        )

        slot = self._prefill_workspaces[infer_state.microbatch_index]
        entry = slot["generic"]
        rows, columns = topk_indices.shape
        if (
            entry is None
            or entry["buffers"]["chunk_indices"].shape[0] < rows
            or entry["buffers"]["chunk_indices"].shape[1] != columns
            or entry["buffers"]["chunk_indices"].dtype != topk_indices.dtype
            or entry["buffers"]["prefill_row_table"].numel() < packed_kv.shape[0]
        ):
            capacity = max(self.model.batch_max_tokens or 0, rows)
            entry = {
                "buffers": allocate_generic_prefill_workspace(q, packed_kv, topk_indices, query_capacity=capacity),
                "owner": None,
            }
            slot["generic"] = entry
        mem_index = infer_state.mem_index
        binding = (id(mem_index), mem_index.data_ptr(), tuple(topk_indices.shape), packed_kv.shape[0])
        if entry["owner"] is None or entry["owner"]() is not infer_state or entry["binding"] != binding:
            buffers = entry["buffers"]
            # Capacity is persistent; the current batch's mapping is not.
            buffers["prefill_row_table"].fill_(-1)
            entry["workspace"] = {
                "prefill_row_table": buffers["prefill_row_table"][: packed_kv.shape[0]],
                "chunk_indices": buffers["chunk_indices"][:rows],
                "chunk_lengths": buffers["chunk_lengths"][:rows],
                "cu_seqlens_q": buffers["cu_seqlens_q"][: rows + 1],
                "row_table_built": False,
            }
            entry["owner"] = weakref.ref(infer_state)
            entry["binding"] = binding
        return entry["workspace"]

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
        )

        params = att_control.nsa_prefill_dict
        # Only H64 uses the small-Q workspace paths.
        if q.shape[1] != 64:
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

        from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
            can_use_direct_smallq,
            can_use_selected_prefill,
        )

        direct_workspace = (
            self.backend.get_smallq_workspace(self.infer_state, q)
            if can_use_direct_smallq(
                q,
                kv,
                self.ragged_mem_index,
                self.infer_state.mem_index,
                params["prefill_cache_kv"],
                params["topk_indices"],
            )
            else None
        )
        selected_eligible = direct_workspace is None and can_use_selected_prefill(
            q,
            kv,
            self.ragged_mem_index,
            self.infer_state.mem_index,
            params["prefill_cache_kv"],
            params["topk_indices"],
        )
        workspace = self.backend.get_smallq_workspace(self.infer_state, q) if selected_eligible else None
        generic_workspace = None
        if direct_workspace is None and not selected_eligible:
            graph_guard = (
                self.infer_state.is_cuda_graph_capture
                or self.backend.model.prefill_graph is not None
                or hasattr(self.infer_state, "prefill_cuda_graph_exe_list")
            )
            if not graph_guard and not torch.cuda.is_current_stream_capturing():
                generic_workspace = self.backend.get_generic_prefill_workspace(
                    self.infer_state, q, kv, params["topk_indices"]
                )
        return context_attention_fwd_fp8(
            q,
            kv,
            self.ragged_mem_index,
            self.infer_state.mem_index,
            params["prefill_cache_kv"],
            self.ks,
            params["topk_indices"],
            params["softmax_scale"],
            workspace,
            direct_workspace,
            generic_workspace,
        )


@dataclasses.dataclass
class Fp8Glm5NextSparseDecodeState(NsaFlashMlaFp8SparseDecodeAttState):
    decode_topk_bound: int = None

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

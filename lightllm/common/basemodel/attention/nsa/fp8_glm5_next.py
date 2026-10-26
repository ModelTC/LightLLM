import dataclasses
import weakref

import torch

from .fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend, NsaFlashMlaFp8SparseDecodeAttState
from .glm5_next import Glm5NextSparsePrefillState


class Fp8Glm5NextSparseAttBackend(NsaFlashMlaFp8SparseAttBackend):
    def create_att_prefill_state(self, infer_state):
        return Fp8Glm5NextSparsePrefillState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state):
        config = getattr(self.model, "config", {})
        # 2048 pooled entries plus at most three kpool remainder entries fit the 2112 rounded direct window.
        bound = (
            2051
            if isinstance(config, dict) and config.get("index_topk") == 2048 and config.get("index_kpool") == 4
            else None
        )
        return Fp8Glm5NextSparseDecodeState(
            backend=self, _infer_state_weak_ref=weakref.ref(infer_state), decode_topk_bound=bound
        )


@dataclasses.dataclass
class Fp8Glm5NextSparsePrefillState(Glm5NextSparsePrefillState):
    def _smallq_workspace(self, q):
        key = (q.shape[0], q.shape[1], q.device, q.dtype)
        cache = getattr(self.infer_state, "_glm5_fp8_smallq_workspace_cache", None)
        if cache is None:
            cache = {}
            self.infer_state._glm5_fp8_smallq_workspace_cache = cache
        workspace = cache.get(key)
        if workspace is None:
            if q.shape[0] == 1:
                from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
                    _allocate_direct_smallq_workspace,
                )

                workspace = _allocate_direct_smallq_workspace(q)
            else:
                from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
                    _allocate_selected_prefill_workspace,
                )

                workspace = _allocate_selected_prefill_workspace(q)
            cache[key] = workspace
        return workspace

    def _generic_prefill_workspace(self, q, packed_kv, topk_indices):
        mem_index = self.infer_state.mem_index
        key = (
            "generic_prefill",
            q.shape[0],
            packed_kv.shape[0],
            q.device,
            q.dtype,
            mem_index.data_ptr(),
            id(mem_index),
            topk_indices.shape,
            topk_indices.dtype,
        )
        cache = getattr(self.infer_state, "_fp8_sparse_generic_prefill_workspace_cache", None)
        if cache is None:
            cache = {}
            self.infer_state._fp8_sparse_generic_prefill_workspace_cache = cache
        workspace = cache.get(key)
        if workspace is None:
            from lightllm.models.glm5_next.triton_kernel.context_flashattention_nopad_fp8 import (
                _allocate_generic_prefill_workspace,
            )

            workspace = _allocate_generic_prefill_workspace(q, packed_kv, topk_indices)
            cache[key] = workspace
        return workspace

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
            _can_use_direct_smallq,
            _can_use_selected_prefill,
        )

        direct_workspace = (
            self._smallq_workspace(q)
            if _can_use_direct_smallq(
                q,
                kv,
                self.ragged_mem_index,
                self.infer_state.mem_index,
                params["prefill_cache_kv"],
                params["topk_indices"],
            )
            else None
        )
        selected_eligible = direct_workspace is None and _can_use_selected_prefill(
            q,
            kv,
            self.ragged_mem_index,
            self.infer_state.mem_index,
            params["prefill_cache_kv"],
            params["topk_indices"],
        )
        workspace = self._smallq_workspace(q) if selected_eligible else None
        generic_workspace = None
        if direct_workspace is None and not selected_eligible:
            model = getattr(self.backend, "model", None)
            graph_guard = (
                getattr(self.infer_state, "is_cuda_graph_capture", False)
                or getattr(model, "prefill_graph", None) is not None
                or hasattr(self.infer_state, "prefill_cuda_graph_exe_list")
            )
            if not graph_guard and not torch.cuda.is_current_stream_capturing():
                generic_workspace = self._generic_prefill_workspace(q, kv, params["topk_indices"])
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
        if fp8_sparse_decode._can_use_fp8_sparse_direct_decode(
            q_nope,
            packed_kv,
            topk_mem_indices,
            params["softmax_scale"],
            params["kv_lora_rank"],
            self.decode_topk_bound,
            is_cuda_graph_capture=getattr(self.infer_state, "is_cuda_graph_capture", False),
        ):
            return fp8_sparse_decode._fp8_sparse_direct_decode(
                q_nope, packed_kv, topk_mem_indices, params["softmax_scale"]
            )
        padded_q = pad_query_heads(
            q_nope, is_cuda_graph_capture=getattr(self.infer_state, "is_cuda_graph_capture", False)
        )
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

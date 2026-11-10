import dataclasses
import weakref

import torch
import triton

from .glm5_next import Glm5NextSparsePrefillState, Glm5NextSparseDecodeState
from ..base_att import BaseAttBackend


# Bound the temporary BF16 gather buffer to 2 GiB (2097152 × 1 × 512 × 2 bytes).
KV_GATHER_CHUNK_SIZE = 2097152


def allocate_prefill_workspace(q, topk_indices, q_token_capacity=None):
    capacity = q.shape[0] if q_token_capacity is None else q_token_capacity
    return {
        "chunk_topk_indices": torch.empty((capacity, topk_indices.shape[1]), dtype=topk_indices.dtype, device=q.device),
        "chunk_cache_seqlens": torch.empty(capacity, dtype=torch.int32, device=q.device),
        "cu_seqlens_q": torch.arange(capacity + 1, dtype=torch.int32, device=q.device),
    }


class Fp8Glm5NextSparseAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model=model)
        self._prefill_workspaces = [None, None]

    def get_prefill_workspace(self, infer_state, q, topk_indices):
        buffers = self._prefill_workspaces[infer_state.microbatch_index]
        q_token_num, topk_size = topk_indices.shape
        if (
            buffers is None
            or buffers["chunk_topk_indices"].shape[0] < q_token_num
            or buffers["chunk_topk_indices"].shape[1] != topk_size
            or buffers["chunk_topk_indices"].dtype != topk_indices.dtype
        ):
            buffers = allocate_prefill_workspace(
                q, topk_indices, q_token_capacity=max(self.model.batch_max_tokens or 0, q_token_num)
            )
            self._prefill_workspaces[infer_state.microbatch_index] = buffers
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
    @torch.no_grad()
    def _nsa_prefill_att(self, q, packed_kv, att_control):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
        from lightllm.models.glm5_next.triton_kernel.fp8_prefill import (
            _gather_prefill_kv_kernel,
            _compact_topk_indices_kernel,
            _merge_att_output_kernel,
        )

        nsa_prefill_dict = att_control.nsa_prefill_dict
        infer_state = self.infer_state
        workspace = None
        graph_guard = (
            infer_state.is_cuda_graph_capture
            or self.backend.model.prefill_graph is not None
            or hasattr(infer_state, "prefill_cuda_graph_exe_list")
        )
        if not graph_guard and not torch.cuda.is_current_stream_capturing():
            workspace = self.backend.get_prefill_workspace(infer_state, q, nsa_prefill_dict["topk_indices"])
        q_token_num, q_head_num, _ = q.shape
        prefill_cache_kv = nsa_prefill_dict["prefill_cache_kv"]
        topk_indices = nsa_prefill_dict["topk_indices"]
        softmax_scale = nsa_prefill_dict["softmax_scale"]
        req_to_token_indexs = infer_state.req_manager.req_to_token_indexs
        b_req_idx = infer_state.b_req_idx
        b1_cu_kv_seq_len = infer_state.b1_cu_kv_seq_len
        b1_cu_q_seq_len = infer_state.b1_cu_q_seq_len
        total_token_num = infer_state.total_token_num
        gather_chunk_size = min(KV_GATHER_CHUNK_SIZE, total_token_num)
        workspace = workspace or allocate_prefill_workspace(q, topk_indices)
        chunk_topk_indices = workspace["chunk_topk_indices"]
        chunk_cache_seqlens = workspace["chunk_cache_seqlens"]
        cu_seqlens_q = workspace["cu_seqlens_q"]
        # Keep this potentially 2 GiB buffer local to a DSA layer.
        has_history = total_token_num > q_token_num
        gathered_kv = q.new_empty((gather_chunk_size, 1, 512)) if has_history else None
        if not has_history and prefill_cache_kv.stride(-1) != 1:
            prefill_cache_kv = prefill_cache_kv.contiguous()
        output = softmax_lse = None
        for chunk_start in range(0, total_token_num, gather_chunk_size):
            chunk_token_num = min(gather_chunk_size, total_token_num - chunk_start)
            if has_history:
                _gather_prefill_kv_kernel[(chunk_token_num // 2 + b_req_idx.numel(),)](
                    packed_kv,
                    prefill_cache_kv,
                    gathered_kv,
                    req_to_token_indexs,
                    b_req_idx,
                    b1_cu_kv_seq_len,
                    b1_cu_q_seq_len,
                    chunk_start,
                    chunk_token_num,
                    packed_kv.stride(0),
                    prefill_cache_kv.stride(0),
                    prefill_cache_kv.stride(2),
                    req_to_token_indexs_stride_b=req_to_token_indexs.stride(0),
                    req_to_token_indexs_stride_s=req_to_token_indexs.stride(1),
                    batch_size=b_req_idx.numel(),
                    BLOCK_ROWS=2,
                    num_warps=2,
                )
                chunk_kv = gathered_kv[:chunk_token_num].view(-1, 1, 1, 512)
            else:
                # Current BF16 KV already has the complete request layout.
                chunk_kv = prefill_cache_kv[chunk_start : chunk_start + chunk_token_num].view(-1, 1, 1, 512)
            _compact_topk_indices_kernel[(q_token_num,)](
                topk_indices,
                self.q_kv_start_loc,
                chunk_topk_indices,
                chunk_cache_seqlens,
                chunk_start,
                chunk_start + chunk_token_num,
                topk_indices.stride(0),
                topk_indices.stride(1),
                TOPK=topk_indices.shape[1],
                BLOCK=triton.next_power_of_2(topk_indices.shape[1]),
            )
            chunk_output, chunk_softmax_lse, *_ = flash_attn_with_kvcache(
                # only_qv ignores Q/K; these views avoid allocating placeholder tensors.
                q=q[..., :64],
                k_cache=chunk_kv[..., :64],
                qv=q,
                v_cache=chunk_kv,
                page_table=chunk_topk_indices,
                cache_seqlens=chunk_cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                softmax_scale=softmax_scale,
                causal=False,
                only_qv=True,
                num_splits=1,
                return_softmax_lse=True,
            )
            if chunk_start == 0:
                output, softmax_lse = chunk_output, chunk_softmax_lse
            else:
                # Keep the online reduction in FP32 across all remaining chunks.
                output = output.float()
                _merge_att_output_kernel[(q_token_num, triton.cdiv(q_head_num, 4))](
                    output,
                    softmax_lse,
                    chunk_output,
                    chunk_softmax_lse,
                    chunk_softmax_lse.stride(1),
                    chunk_softmax_lse.stride(0),
                    q_head_num=q_head_num,
                    q_token_num=q_token_num,
                    BLOCK_HEADS=4,
                )
            del chunk_output, chunk_softmax_lse
        return output.to(q.dtype)


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

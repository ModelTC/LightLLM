from lightllm.common.attention_state.full_att import FullAttentionState
import torch
import triton

from lightllm.utils.vllm_utils import HAS_VLLM, vllm_ops

from .triton_kernel.index_quant import hadamard_transform_quant_fp8
from .triton_kernel.kpool import compress_pools, gather_pools, gather_paged_pools, get_pool_ranges, expand_topk


class Glm5NextNsaInfer:
    """K-pool indexing with pooled token KV and a small per-request raw tail."""

    def __init__(self, layer_idx, network_config, tp_world_size):
        self.layer_idx = layer_idx
        self.index_topk = network_config["index_topk"]
        self.index_n_heads = network_config["index_n_heads"]
        self.index_head_dim = network_config["index_head_dim"]
        self.rms_norm_eps = network_config["rms_norm_eps"]
        self.index_kpool = network_config["index_kpool"]

    def select_topk_indices(self, logits, q_pool_seq_len, topk_pool_indices):
        """Select row-relative indices, with valid entries before -1 padding."""
        if HAS_VLLM and hasattr(vllm_ops, "top_k_per_row_decode"):
            # next_n=1 treats each query as an independent row, including prefill.
            # The decode entry splits long rows before merging their candidates.
            # Tested with vLLM 0.22.1; persistent_topk can drop candidates (#51782).
            vllm_ops.top_k_per_row_decode(
                logits,
                1,
                q_pool_seq_len,
                topk_pool_indices,
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_pool_indices.shape[1],
            )
        else:
            pool_positions = torch.arange(logits.shape[1], device=logits.device)
            logits.masked_fill_(pool_positions[None, :] >= q_pool_seq_len[:, None], -float("inf"))
            selected = torch.topk(logits, topk_pool_indices.shape[1], dim=-1, sorted=True).indices
            topk_pool_indices.copy_(selected.masked_fill(selected >= q_pool_seq_len[:, None], -1))

    def _get_indices(self, hidden_states, q_lora, infer_state, att_state, layer_weight):
        k = layer_weight.k_norm_(layer_weight.wk_proj_.mm(hidden_states), eps=self.rms_norm_eps)
        gate = layer_weight.index_kpool_compress_gate.mm(hidden_states)
        k_and_gate = torch.cat((k, gate), -1)
        att_state_role = infer_state.att_state_role
        indexer_tail = infer_state.att_state_manager.get_state(
            model_role=att_state_role,
            state_type_filter=f"compressor_indexer_w{self.index_kpool}_s{self.index_kpool}",
            logical_layer_id=self.layer_idx,
        ).get_layer_view(logical_layer_id=self.layer_idx)
        full_att_state = infer_state.att_state_manager.get_state(
            model_role=infer_state.att_state_role,
            state_type_filter="mla",
            logical_layer_id=self.layer_idx,
            state_class=FullAttentionState,
        )
        view = full_att_state.get_layer_view(logical_layer_id=self.layer_idx)
        indexer_k = view.indexer_k
        compress_pools(
            k_and_gate=k_and_gate,
            indexer_tail=indexer_tail,
            indexer_k=indexer_k,
            index_kpool_compress_ape=layer_weight.index_kpool_compress_ape.weight,
            q_kv_seq_len=att_state.q_kv_seq_len,
            mem_index=infer_state.mem_index,
            b_req_idx=infer_state.b_req_idx,
            b1_cu_q_seq_len=infer_state.b1_cu_q_seq_len,
            b_seq_len=infer_state.b_seq_len,
            max_q_seq_len=infer_state.max_q_seq_len,
            b_mtp_index=None if infer_state.is_prefill else infer_state.b_mtp_index,
            indexer_token_stride=view.indexer_token_stride,
        )

        if infer_state.max_kv_seq_len <= self.index_topk:
            return expand_topk(
                None,
                att_state.q_kv_seq_len,
                att_state.q_req_idx,
                infer_state.req_manager.req_to_token_indexs,
                self.index_topk,
                dense=True,
            )

        # The small indexer is replicated: no all-gather of query heads and
        # identical pool selection on every TP rank.
        q = layer_weight.wq_b_proj_.mm(q_lora).view(-1, self.index_n_heads, self.index_head_dim)
        q_fp8, q_scale = hadamard_transform_quant_fp8(q, scale=self.index_head_dim ** -0.5)
        weights = layer_weight.weights_proj_.mm(hidden_states.float())
        weights = weights * (self.index_n_heads ** -0.5 * self.index_head_dim ** -0.5) * q_scale.squeeze(-1)
        max_pools = triton.cdiv(infer_state.max_kv_seq_len, 4 * 128) * 128
        # Prefill usually has many queries: parallelize over Q and reuse K within each query tile.
        # Decode has few queries, so paged MQA also splits K across SMs for parallelism.
        if infer_state.is_prefill:
            topk_pool_indices = self._get_prefill_indices(
                q_fp8, weights, indexer_k, infer_state, att_state, max_pools, view.indexer_token_stride
            )
        else:
            topk_pool_indices = self._get_decode_indices(
                q_fp8, weights, indexer_k, infer_state, att_state, max_pools, view.indexer_token_stride
            )
        return expand_topk(
            topk_pool_indices,
            att_state.q_kv_seq_len,
            att_state.q_req_idx,
            infer_state.req_manager.req_to_token_indexs,
            self.index_topk,
        )

    def _get_prefill_chunk_size(self, max_pools):
        # Account for DeepGEMM padding within the 512 MiB workspace.
        query_tile = 128 // self.index_n_heads
        logits_row_bytes = triton.cdiv(max_pools, 256) * 256 * 4
        query_tiles = max(1, 512 * 1024 ** 2 // (query_tile * logits_row_bytes))
        return query_tiles * query_tile

    def _get_prefill_indices(
        self, q_fp8, weights, indexer_k, infer_state, att_state, max_pools, indexer_token_stride=1
    ):
        k_fp8_and_scale = gather_pools(
            indexer_k,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_seq_len,
            max_pools,
            indexer_token_stride=indexer_token_stride,
        )
        ks, ke, q_pool_seq_len = get_pool_ranges(
            att_state.q_kv_seq_len, infer_state.b1_cu_q_seq_len, infer_state.max_q_seq_len, max_pools
        )
        topk_pool_indices = torch.empty((q_fp8.shape[0], self.index_topk // 4), dtype=torch.int32, device=q_fp8.device)
        chunk_size = self._get_prefill_chunk_size(max_pools)
        import deep_gemm

        for start in range(0, q_fp8.shape[0], chunk_size):
            end = min(start + chunk_size, q_fp8.shape[0])
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8[start:end],
                k_fp8_and_scale,
                weights[start:end],
                ks[start:end],
                ke[start:end],
                clean_logits=False,
                max_seqlen_k=max_pools,
            )
            self.select_topk_indices(logits, q_pool_seq_len[start:end], topk_pool_indices[start:end])
            # Free before allocating the next chunk.
            del logits
        return topk_pool_indices

    def _get_decode_indices(self, q_fp8, weights, indexer_k, infer_state, att_state, max_pools, indexer_token_stride=1):
        import deep_gemm

        q_pool_seq_len = (att_state.q_kv_seq_len // 4).view(-1, 1)
        indexer_k_pages, block_table = gather_paged_pools(
            indexer_k,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            q_pool_seq_len,
            max_pools,
            indexer_token_stride=indexer_token_stride,
        )
        mqa_logits_metadata = deep_gemm.get_paged_mqa_logits_metadata(q_pool_seq_len, 64, deep_gemm.get_num_sms())
        # Each MTP position has its own pool length; HOLD rows have no valid pools.
        # Fixed-width mtp_step=2 gives Q=[3 * num_requests, 1, heads, dim], so next_n stays 1.
        # All three verify positions are preserved despite the SM90 kernel's native next_n limit of 2.
        logits = deep_gemm.fp8_paged_mqa_logits(
            q_fp8.unsqueeze(1),
            indexer_k_pages,
            weights,
            q_pool_seq_len,
            block_table,
            mqa_logits_metadata,
            max_pools,
            clean_logits=False,
        )
        topk_pool_indices = torch.empty((q_fp8.shape[0], self.index_topk // 4), dtype=torch.int32, device=q_fp8.device)
        self.select_topk_indices(logits, q_pool_seq_len.view(-1), topk_pool_indices)
        return topk_pool_indices

import deep_gemm
import torch
import triton

from lightllm.common.basemodel import BaseLayerInfer
from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k import destindex_copy_indexer_k
from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_q_indexer_rope_hadamard_quant
from ..triton_kernel.attention import (
    candidate_indexer_logits,
    gather_indexer_k,
    select_indexer_candidates,
    select_indexer_topk,
    select_paged_topk,
)
from lightllm.models.deepseek_v4.triton_kernel.build_compressed_kv_indices import build_compressed_kv_indices


class DeepseekV41IndexInfer(BaseLayerInfer):
    def __init__(self, layer_idx, network_config):
        self.layer_idx_ = layer_idx
        self.compress_ratio = network_config["compress_ratios"][layer_idx]
        self.index_topk = network_config["index_topk"]
        self.index_n_heads = network_config["index_n_heads"]
        self.index_head_dim = network_config["index_head_dim"]
        self.indexer_weight_scale = self.index_head_dim ** -0.5 * self.index_n_heads ** -0.5
        self.eps = network_config["rms_norm_eps"]
        self.is_index_source = layer_idx in network_config["index_source_layer_ids"]
        self.candidate_source = network_config["candidate_source_layer_id"]
        self.candidate_topk_blocks = network_config["candidate_topk_blocks"]
        self.freqs_cis = None

    def write_indexer_k(self, latent, infer_state, layer_weight):
        key = layer_weight.idx_k_norm_(layer_weight.idx_wk_.mm(latent), eps=self.eps)
        destindex_copy_indexer_k(
            key,
            infer_state.mem_index,
            infer_state.position_ids,
            infer_state.get_compressed_kv_view(self.layer_idx_).indexer_k,
            infer_state.get_compressed_kv_state(self.layer_idx_).config.block_token_size // self.compress_ratio,
            compress_ratio=self.compress_ratio,
            freqs_cis=self.freqs_cis,
            round_scale_to_power_of_two=True,
        )

    def build_metadata(self, x, q_lora, infer_state, layer_weight):
        if self.is_index_source:
            infer_state.v41_sparse_selection = self._select_indices(x, q_lora, infer_state, layer_weight)
        indices, lengths = infer_state.v41_sparse_selection if self.compress_ratio else (None, None)
        return {
            "swa_indices": infer_state.dsv4_swa_indices.unsqueeze(1),
            "swa_lengths": infer_state.dsv4_swa_lengths,
            "extra_indices": indices,
            "extra_lengths": lengths,
        }

    def _project_indexer_q(self, x, q_lora, infer_state, layer_weight):
        tokens = q_lora.shape[0]
        query = layer_weight.idx_wq_b_.mm(q_lora).view(tokens, self.index_n_heads, self.index_head_dim)
        weights = self.alloc_tensor((tokens, self.index_n_heads), dtype=torch.float32, device=x.device)
        weights = layer_weight.idx_weights_proj_.mm(x, out=weights)
        query, weights = fused_q_indexer_rope_hadamard_quant(
            query,
            weights,
            self.indexer_weight_scale,
            self.freqs_cis,
            infer_state.position_ids,
            q_fp8=self.alloc_tensor(query.shape, torch.float8_e4m3fn, device=x.device),
            weights_out=self.alloc_tensor((tokens, self.index_n_heads, 1), torch.float32, device=x.device),
            apply_hadamard=False,
            round_scale_to_power_of_two=True,
        )
        return query, weights.squeeze(-1)

    def _select_indices(self, x, q_lora, infer_state, layer_weight):
        ratio = self.compress_ratio
        capacity = triton.cdiv(max(1, infer_state.max_kv_seq_len // ratio), 64) * 64
        candidate_capacity = self.candidate_topk_blocks * 8
        sparse = self.layer_idx_ > self.candidate_source and capacity > candidate_capacity
        build = (
            self.layer_idx_ == self.candidate_source
            and triton.cdiv(infer_state.max_kv_seq_len, 64) * 64 > candidate_capacity
        )
        slots = self.alloc_tensor((x.shape[0], self.index_topk), dtype=torch.int32, device=x.device)
        lengths = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
        if capacity <= self.index_topk and not (build or sparse):
            build_compressed_kv_indices(
                infer_state.dsv4_sparse_req_idx,
                infer_state.position_ids,
                infer_state.req_manager.req_to_token_indexs,
                ratio,
                slots,
                lengths,
                hold_req_id=infer_state.req_manager.HOLD_REQUEST_ID,
                min_length=0,
            )
            return slots.unsqueeze(1), lengths

        query, weights = self._project_indexer_q(x, q_lora, infer_state, layer_weight)
        packed = infer_state.get_compressed_kv_view(self.layer_idx_).indexer_k
        cache = packed.view(packed.shape[0], 64, 1, self.index_head_dim + 4)
        dense = (
            infer_state.is_prefill
            and infer_state.max_q_seq_len > 1
            and max(1, infer_state.total_token_num // ratio) * (self.index_head_dim + 4) <= 256 * 1024 * 1024
        )
        meta = infer_state.get_indexer_metadata(
            self.layer_idx_, capacity, request_pages=infer_state.is_prefill and (dense or sparse)
        )
        if dense and not sparse:
            key_lengths = (infer_state.b_seq_len // ratio).to(torch.int32)
            key_lengths.masked_fill_(infer_state.b_req_idx == infer_state.req_manager.HOLD_REQUEST_ID, 0)
            offsets = key_lengths.cumsum(0, dtype=torch.int32) - key_lengths
            starts = offsets[meta.page_rows.long()]
            ends = starts + meta.valid_lengths
            keys = gather_indexer_k(
                packed,
                ratio,
                infer_state.b_req_idx,
                key_lengths,
                offsets,
                infer_state.req_manager.req_to_token_indexs,
                max(1, infer_state.total_token_num // ratio),
                capacity,
            )
        if build:
            infer_state.candidate_block_indices = torch.empty(
                (x.shape[0], min(self.candidate_topk_blocks, capacity // 8)), dtype=torch.int32, device=x.device
            )
        for start in range(0, x.shape[0], meta.rows_per_chunk):
            rows = slice(start, min(start + meta.rows_per_chunk, x.shape[0]))
            valid = meta.valid_lengths[rows]
            page_rows = meta.page_rows[rows] if meta.page_rows is not None else None
            pages = meta.row_page_table if page_rows is not None else meta.row_page_table[rows]
            candidates = infer_state.candidate_block_indices[rows] if sparse else None
            if sparse:
                logits = candidate_indexer_logits(
                    query[rows], cache, weights[rows], pages, candidates, valid, page_rows=page_rows
                )
            elif dense:
                logits = deep_gemm.fp8_mqa_logits(
                    query[rows],
                    keys,
                    weights[rows],
                    starts[rows],
                    ends[rows],
                    clean_logits=False,
                    max_seqlen_k=capacity,
                )[:, :capacity]
            else:
                logits = deep_gemm.fp8_paged_mqa_logits(
                    query[rows].unsqueeze(1),
                    cache,
                    weights[rows],
                    meta.context_lengths[rows],
                    pages,
                    meta.schedulers[start // meta.rows_per_chunk],
                    capacity,
                    False,
                )[:, :capacity]
            if build:
                select_indexer_candidates(logits, valid, infer_state.candidate_block_indices[rows])
            if self.index_topk != 512 and candidates is None:
                row_pages = pages if page_rows is None else pages[page_rows.long()]
                select_indexer_topk(logits, valid, row_pages, slots[rows], lengths[rows], 64, None, None)
            else:
                select_paged_topk(logits, valid, pages, slots[rows], lengths[rows], candidates, page_rows=page_rows)
        return slots.unsqueeze(1), lengths

import deep_gemm
import torch
import triton

from lightllm.common.basemodel import BaseLayerInfer
from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k import destindex_copy_indexer_k
from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_q_indexer_rope_hadamard_quant
from ..triton_kernel.attention import select_indexer_topk


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
            infer_state.v41_sparse_selection = self._select_indices(infer_state, query, weights.squeeze(-1))
        indices, lengths = infer_state.v41_sparse_selection if self.compress_ratio else (None, None)
        return {
            "swa_indices": infer_state.dsv4_swa_indices.unsqueeze(1),
            "swa_lengths": infer_state.dsv4_swa_lengths,
            "extra_indices": indices,
            "extra_lengths": lengths,
        }

    def _select_indices(self, infer_state, query, weights):
        ratio = self.compress_ratio
        config = infer_state.get_compressed_kv_state(self.layer_idx_).config
        page_size = config.block_token_size // ratio
        packed = infer_state.get_compressed_kv_view(self.layer_idx_).indexer_k
        cache = packed.view(packed.shape[0], page_size, 1, self.index_head_dim + 4)
        capacity = triton.cdiv(max(1, infer_state.max_kv_seq_len // ratio), page_size) * page_size
        meta = infer_state.get_indexer_metadata(self.layer_idx_, capacity)
        slots = self.alloc_tensor((query.shape[0], self.index_topk), dtype=torch.int32, device=query.device)
        lengths = torch.empty_like(meta.valid_lengths)
        if self.layer_idx_ == self.candidate_source:
            infer_state.candidate_block_mask = torch.empty(
                (query.shape[0], capacity // 8), dtype=torch.bool, device=query.device
            )
        for chunk, scheduler in enumerate(meta.schedulers):
            start = chunk * meta.rows_per_chunk
            end = min(start + meta.rows_per_chunk, query.shape[0])
            logits = deep_gemm.fp8_paged_mqa_logits(
                query[start:end].unsqueeze(1),
                cache,
                weights[start:end],
                meta.context_lengths[start:end],
                meta.row_page_table[start:end],
                scheduler,
                capacity,
                False,
            )[:, :capacity]
            candidates = infer_state.candidate_block_mask
            select_indexer_topk(
                logits,
                meta.valid_lengths[start:end],
                meta.row_page_table[start:end],
                slots[start:end],
                lengths[start:end],
                page_size,
                candidates[start:end] if self.layer_idx_ >= self.candidate_source else None,
                self.candidate_topk_blocks if self.layer_idx_ == self.candidate_source else None,
            )
        return slots.unsqueeze(1), lengths

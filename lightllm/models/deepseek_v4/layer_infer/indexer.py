import torch
import triton
import deep_gemm

from lightllm.common.basemodel import BaseLayerInfer
from ..infer_struct import DeepseekV4InferStateInfo
from ..triton_kernel.indexer_topk import indexer_topk
from .compressor import CompressorInfer
from ..triton_kernel.cache_io import gather_indexer_k, get_indexer_k_buffer, get_indexer_k_cache
from ..triton_kernel.destindex_copy_indexer_k import destindex_copy_indexer_k
from ..infer_struct import C4_PREFILL_LOGITS_BUDGET_BYTES


class DeepseekV4IndexInfer(BaseLayerInfer):
    """Build sparse metadata and score C4 compressed KV with the replicated FP8 indexer.

    SWA and C128 indices are prepared once per forward. C4 layers write
    compressed indexer keys, score paged indexer keys, then select up to 512 entries.
    Per-forward page tables and scheduler metadata live on infer_state and are
    reused across C4 layers; this object holds only per-layer configuration.
    """

    def __init__(self, layer_idx: int, network_config: dict, tp_world_size: int):
        self.layer_idx_ = layer_idx
        self.compress_ratio = network_config["compress_ratios"][layer_idx]
        self.index_topk = network_config["index_topk"]
        self.index_head_dim = network_config["index_head_dim"]
        self.index_n_heads = network_config["index_n_heads"]
        self.tp_world_size_ = tp_world_size
        self.freqs_cis = None
        self.indexer_weight_scale = self.index_head_dim ** -0.5 * self.index_n_heads ** -0.5
        self.is_v41 = network_config.get("model_type") == "deepseek_v41"
        if self.is_v41:
            self.eps = network_config["rms_norm_eps"]
            self.is_index_source = layer_idx in network_config["index_source_layer_ids"]
            self.is_candidate_source = layer_idx == network_config["candidate_source_layer_id"]
            self.uses_candidate_blocks = layer_idx > network_config["candidate_source_layer_id"]
            self.candidate_topk_blocks = network_config["candidate_topk_blocks"]
        # C4 layers compress and cache their own indexer keys.
        self.indexer_compressor = (
            CompressorInfer(layer_idx, network_config, tp_world_size, is_in_indexer=True)
            if self.compress_ratio == 4
            else None
        )

    def write_indexer_k(
        self,
        x,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight,
        cos_table,
        sin_table,
        use_custom_tensor_manager=True,
        c4_aux_workspace=None,
    ):
        if self.is_v41:
            # V4.1's KV producer passes its normalized shared latent.
            key = layer_weight.idx_k_norm_(layer_weight.idx_wk_.mm(x), eps=self.eps)
            destindex_copy_indexer_k(
                key,
                infer_state.mem_index,
                infer_state.position_ids,
                get_indexer_k_buffer(infer_state.mem_manager, self.layer_idx_),
                infer_state.req_manager.cache_config.compressed_kv_token_alignment // self.compress_ratio,
                compress_ratio=self.compress_ratio,
                freqs_cis=self.freqs_cis,
            )
            return
        if self.compress_ratio != 4:
            return
        # Only group-end rows in this dense bf16 scratch are valid indexer keys.
        kv_score_out = indexer_k_out = hadamard_out = None
        if c4_aux_workspace is not None:
            kv_score_out, indexer_k_out, hadamard_out = c4_aux_workspace.indexer_k_buffers(x.shape[0])
        scratch = self.indexer_compressor.compress(
            x,
            infer_state,
            layer_weight,
            cos_table,
            sin_table,
            use_custom_tensor_manager=use_custom_tensor_manager,
            kv_score_out=kv_score_out,
            out_buffer=indexer_k_out,
        )
        # Q and K use the same normalized Hadamard transform before FP8 quantization.
        from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import hadamard_transform

        if use_custom_tensor_manager:
            hadamard_out = self.alloc_tensor(scratch.shape, scratch.dtype, device=scratch.device)
        scratch = hadamard_transform(scratch, scale=self.index_head_dim ** -0.5, out=hadamard_out)
        destindex_copy_indexer_k(
            scratch,
            infer_state.mem_index.reshape(-1),
            infer_state.position_ids,
            get_indexer_k_buffer(infer_state.mem_manager, self.layer_idx_),
            infer_state.mem_manager.cache_config.compressed_kv_token_alignment // self.compress_ratio,
        )

    def build_metadata(
        self,
        x,
        q_lora,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight,
        use_custom_tensor_manager=True,
        c4_aux_workspace=None,
    ):
        extra_indices = extra_lengths = None
        if self.is_v41:
            if self.is_index_source:
                infer_state.v41_sparse_selection = self._build_v41_metadata(
                    x, q_lora, infer_state, layer_weight, use_custom_tensor_manager
                )
            if self.compress_ratio:
                extra_indices = infer_state.v41_sparse_selection["extra_indices"]
                extra_lengths = infer_state.v41_sparse_selection["extra_lengths"]
        elif self.compress_ratio == 4:
            idx_q_fp8, weights = self._indexer_q_weight(
                x,
                q_lora,
                infer_state,
                layer_weight,
                use_custom_tensor_manager=use_custom_tensor_manager,
                c4_aux_workspace=c4_aux_workspace,
            )
            extra_indices, extra_lengths = self._c4_indices(infer_state, idx_q_fp8, weights)
        elif self.compress_ratio == 128:
            extra_indices = infer_state.dsv4_c128_indices.unsqueeze(1)
            extra_lengths = infer_state.dsv4_c128_lengths
        return {
            "swa_indices": infer_state.dsv4_swa_indices.unsqueeze(1),
            "swa_lengths": infer_state.dsv4_swa_lengths,
            "extra_indices": extra_indices,
            "extra_lengths": extra_lengths,
        }

    def _indexer_q_weight(
        self,
        x,
        q_lora,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight,
        use_custom_tensor_manager=True,
        c4_aux_workspace=None,
    ):
        # Fuse RoPE, normalized Hadamard and FP8 quantization; fold Q scales into weights.
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope import (
            fused_q_indexer_rope_quant,
        )

        token_num = q_lora.shape[0]
        idx_q_out = raw_w_out = None
        if c4_aux_workspace is not None:
            idx_q_out, raw_w_out = c4_aux_workspace.indexer_q_inputs(token_num)
        elif self.is_v41:
            alloc = self.alloc_tensor if use_custom_tensor_manager else torch.empty
            raw_w_out = alloc((token_num, self.index_n_heads), dtype=torch.float32, device=x.device)
        idx_q = layer_weight.idx_wq_b_.mm(
            q_lora, out=idx_q_out, use_custom_tensor_mananger=use_custom_tensor_manager
        ).view(token_num, self.index_n_heads, self.index_head_dim)
        raw_w = layer_weight.idx_weights_proj_.mm(
            x, out=raw_w_out, use_custom_tensor_mananger=use_custom_tensor_manager
        ).view(token_num, self.index_n_heads)
        idx_q_fp8_out = weights_out = None
        if c4_aux_workspace is not None:
            idx_q_fp8_out, weights_out = c4_aux_workspace.indexer_q_outputs(token_num)
        elif use_custom_tensor_manager:
            idx_q_fp8_out = self.alloc_tensor(idx_q.shape, torch.float8_e4m3fn, device=idx_q.device)
            weights_out = self.alloc_tensor((*idx_q.shape[:-1], 1), torch.float32, device=idx_q.device)
        idx_q_fp8, weights = fused_q_indexer_rope_quant(
            idx_q,
            raw_w,
            self.indexer_weight_scale,
            self.freqs_cis,
            infer_state.position_ids,
            q_fp8=idx_q_fp8_out,
            weights_out=weights_out,
            apply_hadamard=not self.is_v41,
        )  # fp8 [T,H,d]; weights [T,H,1] with q-scale + weight_scale folded
        return idx_q_fp8, weights.squeeze(-1)

    def _c4_indices(self, infer_state: DeepseekV4InferStateInfo, idx_q_fp8, weights):
        """Return all closed C4 entries for short contexts, or select TopK by paged scoring."""
        index_topk = self.index_topk
        max_entries = max(1, infer_state.max_kv_seq_len // 4)
        c4_cap = ((max_entries + 63) // 64) * 64
        infer_state.init_c4_indices(c4_cap if max_entries <= index_topk else index_topk)

        if max_entries <= index_topk:
            from ..triton_kernel.build_compressed_kv_indices import build_compressed_kv_indices

            slots, lengths = infer_state.dsv4_c4_indices, infer_state.dsv4_c4_lengths
            build_compressed_kv_indices(
                infer_state.dsv4_sparse_req_idx,
                infer_state.position_ids,
                infer_state.req_manager.req_to_token_indexs,
                4,
                slots,
                lengths,
            )
            return slots.unsqueeze(1), lengths

        paged_meta = infer_state.get_indexer_metadata(self.layer_idx_, c4_cap)
        mem_manager = infer_state.mem_manager
        page_size = mem_manager.cache_config.compressed_kv_token_alignment // self.compress_ratio
        packed = get_indexer_k_buffer(mem_manager, self.layer_idx_)
        indexer_k_cache = packed.view(
            packed.shape[0],
            page_size,
            1,
            self.index_head_dim + 4,
        )
        top_slots = infer_state.dsv4_c4_indices
        for chunk_idx, scheduler in enumerate(paged_meta.schedulers):
            start = chunk_idx * paged_meta.rows_per_chunk
            end = min(start + paged_meta.rows_per_chunk, idx_q_fp8.shape[0])
            logits = deep_gemm.fp8_paged_mqa_logits(
                idx_q_fp8[start:end].unsqueeze(1),
                indexer_k_cache,
                weights[start:end],
                paged_meta.context_lengths[start:end],
                paged_meta.logits_page_table[start:end],
                scheduler,
                c4_cap,
                False,
            )
            indexer_topk(
                logits,
                paged_meta.valid_lengths[start:end],
                paged_meta.row_page_table[start:end],
                top_slots[start:end],
                page_size,
            )
            del logits  # Release this chunk before allocating the next one.
        return top_slots.unsqueeze(1), infer_state.dsv4_c4_lengths

    def _use_dense_prefill(self, infer_state):
        return (
            self.is_v41
            and infer_state.is_prefill
            and infer_state.max_q_seq_len > 1
            and max(1, infer_state.total_token_num // self.compress_ratio)
            * infer_state.mem_manager.cache_config.indexer_bytes
            <= C4_PREFILL_LOGITS_BUDGET_BYTES
        )

    def _build_v41_metadata(self, x, q_lora, infer_state, layer_weight, use_custom_tensor_manager):
        from lightllm.models.deepseek_v41.triton_kernel.attention import (
            mask_indexer_candidates,
            select_indexer_candidates,
            select_indexer_topk,
        )

        source = infer_state.req_manager.cache_config.layer_to_kv_source[self.layer_idx_]
        cache = get_indexer_k_cache(infer_state.mem_manager, source)
        query, weights = self._indexer_q_weight(
            x, q_lora, infer_state, layer_weight, use_custom_tensor_manager=use_custom_tensor_manager
        )
        capacity = triton.cdiv(max(1, infer_state.max_kv_seq_len // self.compress_ratio), 64) * 64
        dense_prefill = self._use_dense_prefill(infer_state)
        metadata = infer_state.get_indexer_metadata(self.layer_idx_, capacity, dense_prefill=dense_prefill)
        if dense_prefill:
            key_lengths = (infer_state.b_seq_len // self.compress_ratio).to(torch.int32)
            key_lengths.masked_fill_(infer_state.b_req_idx == infer_state.req_manager.HOLD_REQUEST_ID, 0)
            key_offsets = key_lengths.cumsum(0, dtype=torch.int32) - key_lengths
            key_starts = key_offsets[infer_state._dsv4_token_to_batch_idx.long()]
            key_ends = key_starts + metadata.valid_lengths
            keys = gather_indexer_k(
                infer_state.mem_manager,
                source,
                infer_state.b_req_idx,
                key_lengths,
                key_offsets,
                infer_state.req_manager.req_to_token_indexs,
                max(1, infer_state.total_token_num // self.compress_ratio),
                capacity,
            )
        alloc = self.alloc_tensor if use_custom_tensor_manager else torch.empty
        slots = alloc((x.shape[0], self.index_topk), dtype=torch.int32, device=x.device)
        lengths = alloc((x.shape[0],), dtype=torch.int32, device=x.device)
        if self.is_candidate_source:
            infer_state.candidate_block_mask = torch.zeros(
                (x.shape[0], triton.cdiv(capacity, 8)), dtype=torch.bool, device=x.device
            )
        chunk_size = metadata.rows_per_chunk
        for chunk, start in enumerate(range(0, x.shape[0], chunk_size)):
            end = min(start + chunk_size, x.shape[0])
            if dense_prefill:
                logits = deep_gemm.fp8_mqa_logits(
                    query[start:end],
                    keys,
                    weights[start:end],
                    key_starts[start:end],
                    key_ends[start:end],
                    clean_logits=False,
                    max_seqlen_k=capacity,
                )
            else:
                scheduler = metadata.schedulers[chunk]
                logits = deep_gemm.fp8_paged_mqa_logits(
                    query[start:end].unsqueeze(1),
                    cache,
                    weights[start:end],
                    metadata.context_lengths[start:end],
                    metadata.logits_page_table[start:end],
                    scheduler,
                    capacity,
                    False,
                )
            valid = metadata.valid_lengths[start:end]
            if self.is_candidate_source:
                select_indexer_candidates(
                    logits, valid, infer_state.candidate_block_mask[start:end], self.candidate_topk_blocks
                )
            elif self.uses_candidate_blocks:
                mask_indexer_candidates(logits, valid, infer_state.candidate_block_mask[start:end])
            select_indexer_topk(logits, valid, metadata.row_page_table[start:end], slots[start:end], lengths[start:end])
            del logits
        return {"extra_indices": slots.unsqueeze(1), "extra_lengths": lengths}

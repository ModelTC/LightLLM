import torch
import triton
import deep_gemm

from lightllm.common.basemodel import BaseLayerInfer
from ..infer_struct import PagedIndexerMetadata, DeepseekV4InferStateInfo
from ..triton_kernel.topk_transform import topk_transform
from ..triton_kernel.cache_io import INDEXER_PAGE_SIZE, gather_indexer_k, get_indexer_k_buffer, get_indexer_k_cache
from ..triton_kernel.destindex_copy_indexer_k_dsv4 import destindex_copy_indexer_k_dsv4
from ..workspace import C4_LOGITS_ALIGNMENT, C4_PREFILL_LOGITS_BUDGET_BYTES, compressed_kv_capacity
from .compressor import CompressorInfer


class DeepseekV4IndexInfer(BaseLayerInfer):
    """Build sparse metadata for V4 C4/C128 and V4.1 shared CSA2 history.

    SWA and C128 indices are prepared once per forward. C4 layers write
    compressed indexer keys, score paged history, then select up to 512 entries.
    Per-forward page tables and scheduler metadata live on infer_state and are
    reused across C4 layers; this object holds only per-layer configuration.
    V4.1 index sources select shared history, optionally restricted to candidate blocks.
    """

    def __init__(self, layer_idx: int, network_config: dict, tp_world_size: int):
        self.layer_idx_ = layer_idx
        self.compress_ratio = network_config["compress_ratios"][layer_idx]
        self.index_topk = network_config["index_topk"]
        self.index_head_dim = network_config["index_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.index_n_heads = network_config["index_n_heads"]
        self.tp_world_size_ = tp_world_size
        self.freqs_cis = None
        self.indexer_score_scale = self.index_head_dim ** -0.5
        self.indexer_weight_scale = self.indexer_score_scale * self.index_n_heads ** -0.5
        self.is_v41 = network_config.get("model_type") == "deepseek_v41"
        if self.is_v41:
            self.eps = network_config["rms_norm_eps"]
            self.is_index_source = layer_idx in network_config["index_source_layer_ids"]
            self.is_candidate_source = layer_idx == network_config["candidate_source_layer_id"]
            self.uses_candidate_blocks = layer_idx > network_config["candidate_source_layer_id"]
            self.candidate_topk_blocks = network_config["candidate_topk_blocks"]
        # c4 layers own a second compressor (is_in_indexer) that writes the Lightning-Indexer key
        # pool every step; _c4_indices gathers it back + scores via deep_gemm.fp8_mqa_logits.
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
            destindex_copy_indexer_k_dsv4(
                key,
                infer_state.mem_index,
                infer_state.position_ids,
                get_indexer_k_buffer(infer_state.mem_manager, self.layer_idx_),
                infer_state.req_manager.cache_config.history_token_alignment // self.compress_ratio,
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
        # Rotate K (post norm+rope) by the SAME 1/sqrt(d) Hadamard the q kernel applies, so
        # (Hq)·(Hk)=q·k (H orthogonal) and the fp8 quant of K stays accurate.
        from lightllm.models.deepseek3_2.triton_kernel.hadamard_transform import hadamard_transform

        if use_custom_tensor_manager:
            hadamard_out = self.alloc_tensor(scratch.shape, scratch.dtype, device=scratch.device)
        scratch = hadamard_transform(scratch, scale=self.index_head_dim ** -0.5, out=hadamard_out)
        destindex_copy_indexer_k_dsv4(
            scratch,
            infer_state.mem_index.reshape(-1),
            infer_state.position_ids,
            get_indexer_k_buffer(infer_state.mem_manager, self.layer_idx_),
            infer_state.req_manager.cache_config.history_token_alignment // self.compress_ratio,
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
        swa_indices = infer_state.dsv4_swa_indices.unsqueeze(1)
        swa_lengths = infer_state.dsv4_swa_lengths
        positions = infer_state.position_ids
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
            extra_indices, extra_lengths = self._c4_indices(
                infer_state, idx_q_fp8, weights, positions, c4_aux_workspace=c4_aux_workspace
            )
        elif self.compress_ratio == 128:
            extra_indices = infer_state.dsv4_c128_indices.unsqueeze(1)
            extra_lengths = infer_state.dsv4_c128_lengths
        return {
            "swa_indices": swa_indices,
            "swa_lengths": swa_lengths,
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
        # Fused: wq_b mm -> rope(last rope dims) -> 1/sqrt(d) Hadamard -> per-token fp8 quant, with the
        # per-token q scale + indexer_weight_scale folded into weights, all in ONE kernel (was 4 kernels:
        # rotary_emb_fwd + hadamard_transform + act_quant + weights mul). freqs_cis is the compress rope
        # table (same one the main compress-layer Q path uses); positions indexed inside the kernel.
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope import (
            fused_q_indexer_rope_quant,
        )

        token_num = q_lora.shape[0]
        if x.shape[0] != token_num:
            raise RuntimeError(
                f"DeepSeek-V4 indexer expects full-token hidden states, got x={x.shape[0]} q_lora={token_num}"
            )
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
        ).view(
            token_num, self.index_n_heads
        )  # [T, H] raw
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
        capacity = compressed_kv_capacity(infer_state.v41_index_capacity, self.compress_ratio)
        metadata = self._build_indexer_metadata(infer_state, capacity)
        dense_prefill = self._use_dense_prefill(infer_state)
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
        chunk_size = metadata.rows_per_chunk or x.shape[0]
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
                scheduler = (
                    metadata.scheduler if metadata.chunk_schedulers is None else metadata.chunk_schedulers[chunk]
                )
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
        return {"extra_indices": slots.unsqueeze(1), "extra_lengths": lengths}

    def _c4_indices(self, infer_state: DeepseekV4InferStateInfo, idx_q_fp8, weights, positions, c4_aux_workspace=None):
        """c4 scorer via the page-safe deep_gemm.fp8_paged_mqa_logits over the paged c4 indexer pool,
        then masked topk-512 -> c4 slots. Fixed shapes (c4_cap pinned per graph bucket) keep the decode
        cuda graph capturable."""
        workspace = infer_state.dsv4_workspace
        index_topk = self.index_topk
        max_entries = max(1, int(infer_state.max_kv_seq_len) // 4)
        c4_cap = compressed_kv_capacity(infer_state.max_kv_seq_len, 4)

        if max_entries <= index_topk:
            from ..triton_kernel.build_compress_index_dsv4 import build_compress_index

            slots, lengths = workspace.c4(infer_state.microbatch_index, positions.shape[0], c4_cap)
            slots, lengths = build_compress_index(
                infer_state.dsv4_sparse_req_idx,
                positions,
                infer_state.req_manager.req_to_token_indexs,
                4,
                slots,
                lengths,
            )
            return slots.unsqueeze(1), lengths

        paged_meta = self._build_indexer_metadata(infer_state, c4_cap, c4_aux_workspace)
        return self._score_c4_indices(infer_state, idx_q_fp8, weights, c4_cap, paged_meta)

    def _build_indexer_metadata(self, infer_state: DeepseekV4InferStateInfo, capacity, c4_aux_workspace=None):
        """Build once per forward; captured producers refresh these buffers on graph replay."""
        if (
            infer_state.indexer_metadata is not None
            and infer_state.indexer_metadata.compress_ratio == self.compress_ratio
        ):
            return infer_state.indexer_metadata
        mem_manager = infer_state.mem_manager
        page_size = INDEXER_PAGE_SIZE
        positions = infer_state.position_ids
        from ..triton_kernel.gather_c4_indexer_k_dsv4 import build_indexer_page_table

        b_req_idx = infer_state.b_req_idx
        batch = b_req_idx.shape[0]
        token_num = positions.numel()
        if c4_aux_workspace is None:
            history_lengths = torch.div(infer_state.b_seq_len, self.compress_ratio, rounding_mode="floor").to(
                torch.int32
            )
            page_table_out = row_page_table = None
        else:
            page_cap = capacity // page_size
            page_table_out, row_page_table = c4_aux_workspace.page_tables(batch, token_num, page_cap)
            history_lengths, _, _, _ = c4_aux_workspace.metadata(batch, token_num)
            history_lengths.copy_(torch.div(infer_state.b_seq_len, self.compress_ratio, rounding_mode="floor"))
        page_table = build_indexer_page_table(
            mem_manager,
            b_req_idx,
            history_lengths,
            capacity,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.req_manager.HOLD_REQUEST_ID,
            out=page_table_out,
            compress_ratio=self.compress_ratio,
        )

        if infer_state.is_prefill:
            token_batch_pos = infer_state._dsv4_token_to_batch_idx
            if row_page_table is None:
                row_page_table = page_table[token_batch_pos]
            else:
                torch.index_select(page_table, 0, token_batch_pos, out=row_page_table)
        else:
            row_page_table = page_table

        if c4_aux_workspace is None:
            valid_len = ((positions + 1) // self.compress_ratio).to(torch.int32)
            ctx_lens = torch.clamp(valid_len, min=1).reshape(-1, 1)
            topk_lengths = torch.clamp(torch.minimum(valid_len, torch.full_like(valid_len, self.index_topk)), min=1)
        else:
            _, valid_len, ctx_lens, topk_lengths = c4_aux_workspace.metadata(batch, token_num)
            valid_len.copy_((positions + 1) // self.compress_ratio)
            torch.clamp(valid_len, min=1, out=ctx_lens[:, 0])
            torch.clamp(valid_len, min=1, max=self.index_topk, out=topk_lengths)
        if self.is_v41:
            valid_len.masked_fill_(infer_state.dsv4_sparse_req_idx == infer_state.req_manager.HOLD_REQUEST_ID, 0)
            ctx_lens = valid_len.clamp_min(1).reshape(-1, 1)
        logits_page_table = row_page_table
        entries = mem_manager.cache_config.history_token_alignment // self.compress_ratio
        if entries > page_size:
            source = infer_state.req_manager.cache_config.layer_to_kv_source[self.layer_idx_]
            packed = get_indexer_k_buffer(mem_manager, source)
            slabs = entries // page_size
            logits_page_table = (row_page_table // slabs) * (
                packed.stride(0) // mem_manager.cache_config.indexer_head_dim
            ) + (row_page_table % slabs) * (
                page_size * mem_manager.cache_config.indexer_bytes // mem_manager.cache_config.indexer_head_dim
            )
        dense_prefill = self._use_dense_prefill(infer_state)
        rows_per_chunk = None
        chunk_metadata = None
        if infer_state.is_prefill:
            if c4_aux_workspace is None:
                aligned_capacity = ((capacity + C4_LOGITS_ALIGNMENT - 1) // C4_LOGITS_ALIGNMENT) * C4_LOGITS_ALIGNMENT
                rows_per_chunk = max(1, C4_PREFILL_LOGITS_BUDGET_BYTES // (aligned_capacity * 4))
            else:
                rows_per_chunk = c4_aux_workspace.rows_per_logits_chunk(capacity)
            if positions.numel() > rows_per_chunk and not dense_prefill:
                chunk_metadata = tuple(
                    deep_gemm.get_paged_mqa_logits_metadata(
                        ctx_lens[start : start + rows_per_chunk],
                        page_size,
                        deep_gemm.get_num_sms(),
                    )
                    for start in range(0, positions.numel(), rows_per_chunk)
                )

        metadata = (
            deep_gemm.get_paged_mqa_logits_metadata(
                ctx_lens,
                page_size,
                deep_gemm.get_num_sms(),
            )
            if chunk_metadata is None and not dense_prefill
            else None
        )
        infer_state.indexer_metadata = PagedIndexerMetadata(
            compress_ratio=self.compress_ratio,
            logits_page_table=logits_page_table,
            row_page_table=row_page_table,
            valid_lengths=valid_len,
            context_lengths=ctx_lens,
            scheduler=metadata,
            topk_lengths=topk_lengths,
            rows_per_chunk=rows_per_chunk,
            chunk_schedulers=chunk_metadata,
        )
        return infer_state.indexer_metadata

    def _score_c4_indices(self, infer_state, idx_q_fp8, weights, c4_cap, paged_meta: PagedIndexerMetadata):
        mem_manager = infer_state.mem_manager
        workspace = infer_state.dsv4_workspace
        page_size = INDEXER_PAGE_SIZE
        index_topk = self.index_topk
        indexer_k_cache = get_indexer_k_cache(mem_manager, self.layer_idx_)
        top_slots, _ = workspace.c4(infer_state.microbatch_index, idx_q_fp8.shape[0], index_topk)
        if paged_meta.chunk_schedulers is not None:
            for chunk_idx, start in enumerate(range(0, idx_q_fp8.shape[0], paged_meta.rows_per_chunk)):
                end = min(start + paged_meta.rows_per_chunk, idx_q_fp8.shape[0])
                self._c4_score_topk(
                    idx_q_fp8[start:end],
                    indexer_k_cache,
                    weights[start:end],
                    paged_meta.context_lengths[start:end],
                    paged_meta.row_page_table[start:end],
                    paged_meta.chunk_schedulers[chunk_idx],
                    c4_cap,
                    paged_meta.valid_lengths[start:end],
                    top_slots[start:end],
                    page_size,
                )
            return top_slots.unsqueeze(1), paged_meta.topk_lengths

        self._c4_score_topk(
            idx_q_fp8,
            indexer_k_cache,
            weights,
            paged_meta.context_lengths,
            paged_meta.row_page_table,
            paged_meta.scheduler,
            c4_cap,
            paged_meta.valid_lengths,
            top_slots,
            page_size,
        )
        return top_slots.unsqueeze(1), paged_meta.topk_lengths

    @staticmethod
    def _c4_score_topk(
        idx_q_fp8,
        indexer_k_cache,
        weights,
        ctx_lens,
        row_page_table,
        metadata,
        c4_cap,
        valid_len,
        top_slots,
        page_size,
    ):
        logits = deep_gemm.fp8_paged_mqa_logits(
            idx_q_fp8.unsqueeze(1),
            indexer_k_cache,
            weights,
            ctx_lens,
            row_page_table,
            metadata,
            c4_cap,
            False,
        )
        topk_transform(
            logits,
            valid_len,
            row_page_table,
            top_slots,
            page_size,
        )

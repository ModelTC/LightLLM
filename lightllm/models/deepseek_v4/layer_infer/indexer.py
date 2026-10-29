import torch
import deep_gemm

from lightllm.common.basemodel import BaseLayerInfer
from ..infer_struct import DeepseekV4InferStateInfo
from ..triton_kernel.indexer_topk import indexer_topk
from .compressor import CompressorInfer


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
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import get_indexer_k_buffer
        from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k import (
            destindex_copy_indexer_k,
        )

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
        if self.compress_ratio == 4:
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
            fused_q_indexer_rope_hadamard_quant,
        )

        token_num = q_lora.shape[0]
        idx_q_out = raw_w_out = None
        if c4_aux_workspace is not None:
            idx_q_out, raw_w_out = c4_aux_workspace.indexer_q_inputs(token_num)
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
        idx_q_fp8, weights = fused_q_indexer_rope_hadamard_quant(
            idx_q,
            raw_w,
            self.indexer_weight_scale,
            self.freqs_cis,
            infer_state.position_ids,
            q_fp8=idx_q_fp8_out,
            weights_out=weights_out,
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

        paged_meta = infer_state.get_c4_indexer_metadata(c4_cap)
        mem_manager = infer_state.mem_manager
        page_size = mem_manager.cache_config.compressed_kv_token_alignment // self.compress_ratio
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import get_indexer_k_buffer

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
                paged_meta.row_page_table[start:end],
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

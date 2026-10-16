"""CUDA bindings to existing kernels; algorithms and signatures are unchanged."""

KERNEL_BINDINGS = {
    "attention.Fa3AttBackend": (
        "lightllm.common.basemodel.attention.fa3.fp",
        "Fa3AttBackend",
    ),
    "attention.FlashInferAttBackend": (
        "lightllm.common.basemodel.attention.flashinfer.fp",
        "FlashInferAttBackend",
    ),
    "attention.Fp8Fa3AttBackend": (
        "lightllm.common.basemodel.attention.fa3.fp8",
        "Fp8Fa3AttBackend",
    ),
    "attention.Fp8FlashInferAttBackend": (
        "lightllm.common.basemodel.attention.flashinfer.fp8",
        "Fp8FlashInferAttBackend",
    ),
    "attention.Int4kvTritonAttBackend": (
        "lightllm.common.basemodel.attention.triton.int4kv",
        "Int4kvTritonAttBackend",
    ),
    "attention.Int8kvTritonAttBackend": (
        "lightllm.common.basemodel.attention.triton.int8kv",
        "Int8kvTritonAttBackend",
    ),
    "attention.MlaFa3AttBackend": (
        "lightllm.common.basemodel.attention.fa3.mla",
        "MlaFa3AttBackend",
    ),
    "attention.MlaFlashInferAttBackend": (
        "lightllm.common.basemodel.attention.flashinfer.mla",
        "MlaFlashInferAttBackend",
    ),
    "attention.MlaTritonAttBackend": (
        "lightllm.common.basemodel.attention.triton.mla",
        "MlaTritonAttBackend",
    ),
    "attention.NeoFa3AttBackend": (
        "lightllm.common.basemodel.attention.fa3.neo",
        "NeoFa3AttBackend",
    ),
    "attention.NeoTritonAttBackend": (
        "lightllm.common.basemodel.attention.triton.neo",
        "NeoTritonAttBackend",
    ),
    "attention.NsaFlashMlaFp8SparseAttBackend": (
        "lightllm.common.basemodel.attention.nsa.fp8_flashmla_sparse",
        "NsaFlashMlaFp8SparseAttBackend",
    ),
    "attention.NsaFlashMlaSparseAttBackend": (
        "lightllm.common.basemodel.attention.nsa.flashmla_sparse",
        "NsaFlashMlaSparseAttBackend",
    ),
    "attention.TritonAttBackend": (
        "lightllm.common.basemodel.attention.triton.fp",
        "TritonAttBackend",
    ),
    "attention.linear.FlashQlaLinearAttBackend": (
        "lightllm.common.basemodel.attention.linear.flashqla",
        "FlashQlaLinearAttBackend",
    ),
    "attention.linear.LinearAttBackend": (
        "lightllm.common.basemodel.attention.linear.gdn",
        "LinearAttBackend",
    ),
    "attention.linear.LinearAttDecodeAttState": (
        "lightllm.common.basemodel.attention.linear.gdn",
        "LinearAttDecodeAttState",
    ),
    "attention.linear.LinearAttPrefillAttState": (
        "lightllm.common.basemodel.attention.linear.gdn",
        "LinearAttPrefillAttState",
    ),
    "attention.linear.TritonLinearAttBackend": (
        "lightllm.common.basemodel.attention.linear.triton",
        "TritonLinearAttBackend",
    ),
    "common.basemodel.build_chained_mtp_decode_input.build_chained_mtp_decode_input_inplace": (
        "lightllm.common.basemodel.triton_kernel.build_chained_mtp_decode_input",
        "build_chained_mtp_decode_input_inplace",
    ),
    "common.basemodel.copy_kv_index_to_req.copy_kv_index_to_req_prefill": (
        "lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req",
        "copy_kv_index_to_req_prefill",
    ),
    "common.basemodel.copy_kv_index_to_req.select_kv_index_from_req": (
        "lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req",
        "select_kv_index_from_req",
    ),
    "common.basemodel.copy_kv_index_to_req.select_kv_index_from_req_prefill": (
        "lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req",
        "select_kv_index_from_req_prefill",
    ),
    "common.basemodel.destindex_copy_kv.destindex_copy_kv": (
        "lightllm.common.basemodel.triton_kernel.destindex_copy_kv",
        "destindex_copy_kv",
    ),
    "common.basemodel.destindex_copy_kv_fp8.destindex_copy_kv_fp8": (
        "lightllm.common.basemodel.triton_kernel.destindex_copy_kv_fp8",
        "destindex_copy_kv_fp8",
    ),
    "common.basemodel.dynamic_mtp_utils.prepare_dynamic_mtp_model_input": (
        "lightllm.common.basemodel.triton_kernel.dynamic_mtp_utils",
        "prepare_dynamic_mtp_model_input",
    ),
    "common.basemodel.embedding.embedding": (
        "lightllm.common.basemodel.triton_kernel.embedding",
        "embedding",
    ),
    "common.basemodel.fused_moe.grouped_fused_moe_ep.use_sm100_mega_moe": (
        "lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep",
        "use_sm100_mega_moe",
    ),
    "common.basemodel.fused_moe.moe_silu_and_mul.silu_and_mul_fwd": (
        "lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul",
        "silu_and_mul_fwd",
    ),
    "common.basemodel.gather_token_id.gather_token": (
        "lightllm.common.basemodel.triton_kernel.gather_token_id",
        "gather_token",
    ),
    "common.basemodel.gather_token_id.gather_token_prefill_decode_mixed": (
        "lightllm.common.basemodel.triton_kernel.gather_token_id",
        "gather_token_prefill_decode_mixed",
    ),
    "common.basemodel.gather_token_id.scatter_token": (
        "lightllm.common.basemodel.triton_kernel.gather_token_id",
        "scatter_token",
    ),
    "common.basemodel.gen_decode_params.gen_decode_params": (
        "lightllm.common.basemodel.triton_kernel.gen_decode_params",
        "gen_decode_params",
    ),
    "common.basemodel.gen_mtp_prefill_params.gen_mtp_new_input_ids": (
        "lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params",
        "gen_mtp_new_input_ids",
    ),
    "common.basemodel.gen_prefill_params.gen_prefill_params": (
        "lightllm.common.basemodel.triton_kernel.gen_prefill_params",
        "gen_prefill_params",
    ),
    "common.basemodel.gen_sampling_params.token_id_counter": (
        "lightllm.common.basemodel.triton_kernel.gen_sampling_params",
        "token_id_counter",
    ),
    "common.basemodel.gen_sampling_params.update_req_to_token_id_counter": (
        "lightllm.common.basemodel.triton_kernel.gen_sampling_params",
        "update_req_to_token_id_counter",
    ),
    "common.basemodel.kv_cache_offload.load_cpu_kv_to_gpu": (
        "lightllm.common.basemodel.triton_kernel.kv_cache_offload",
        "load_cpu_kv_to_gpu",
    ),
    "common.basemodel.kv_cache_offload.offload_gpu_kv_to_cpu": (
        "lightllm.common.basemodel.triton_kernel.kv_cache_offload",
        "offload_gpu_kv_to_cpu",
    ),
    "common.basemodel.kv_copy.mla_copy_kv.destindex_copy_kv": (
        "lightllm.common.basemodel.triton_kernel.kv_copy.mla_copy_kv",
        "destindex_copy_kv",
    ),
    "common.basemodel.kv_copy.ppl_int4kv_copy_kv.destindex_copy_int4kv": (
        "lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv",
        "destindex_copy_int4kv",
    ),
    "common.basemodel.kv_copy.ppl_int8kv_copy_kv.destindex_copy_quantize_kv": (
        "lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int8kv_copy_kv",
        "destindex_copy_quantize_kv",
    ),
    "common.basemodel.kv_move.copy_kv_buffer_to_kv_buffer": (
        "lightllm.common.basemodel.triton_kernel.kv_move",
        "copy_kv_buffer_to_kv_buffer",
    ),
    "common.basemodel.linear_att.causal_conv1d.causal_conv1d_fn": (
        "lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d",
        "causal_conv1d_fn",
    ),
    "common.basemodel.linear_att.causal_conv1d.causal_conv1d_update": (
        "lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d",
        "causal_conv1d_update",
    ),
    "common.basemodel.linear_att.causal_conv1d_mtp.causal_conv1d_update": (
        "lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp",
        "causal_conv1d_update",
    ),
    "common.basemodel.linear_att.fla.ops.chunk_gated_delta_rule": (
        "lightllm.common.basemodel.triton_kernel.linear_att.fla.ops",
        "chunk_gated_delta_rule",
    ),
    "common.basemodel.linear_att.fla.ops.fused_recurrent_gated_delta_rule": (
        "lightllm.common.basemodel.triton_kernel.linear_att.fla.ops",
        "fused_recurrent_gated_delta_rule",
    ),
    "common.basemodel.linear_att.fla.ops.kda.chunk_kda_with_fused_gate": (
        "lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda",
        "chunk_kda_with_fused_gate",
    ),
    "common.basemodel.linear_att.fla.ops.kda_decode.fused_recurrent_kda": (
        "lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda_decode",
        "fused_recurrent_kda",
    ),
    "common.basemodel.linear_att.fused_gdn_gating.fused_gdn_gating": (
        "lightllm.common.basemodel.triton_kernel.linear_att.fused_gdn_gating",
        "fused_gdn_gating",
    ),
    "common.basemodel.linear_att.gdn_decode_pack.conv_pack_gdn_decode_inputs": (
        "lightllm.common.basemodel.triton_kernel.linear_att.gdn_decode_pack",
        "conv_pack_gdn_decode_inputs",
    ),
    "common.basemodel.linear_att.mtp_fused_recurrent.mtp_fused_recurrent_gated_delta_rule": (
        "lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent",
        "mtp_fused_recurrent_gated_delta_rule",
    ),
    "common.basemodel.linear_att.mtp_state_params.build_dynamic_mtp_linear_att_state_params": (
        "lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params",
        "build_dynamic_mtp_linear_att_state_params",
    ),
    "common.basemodel.linear_att_copy.copy_linear_att_state_to_kv_buffer": (
        "lightllm.common.basemodel.triton_kernel.linear_att_copy",
        "copy_linear_att_state_to_kv_buffer",
    ),
    "common.basemodel.linear_att_cpu_cache_copy.copy_cpu_cache_to_kv_buffer": (
        "lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy",
        "copy_cpu_cache_to_kv_buffer",
    ),
    "common.basemodel.linear_att_cpu_cache_copy.copy_kv_buffer_to_cpu_cache": (
        "lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy",
        "copy_kv_buffer_to_cpu_cache",
    ),
    "common.basemodel.linear_att_cpu_cache_copy.copy_linear_att_state_to_linear_att_state": (
        "lightllm.common.basemodel.triton_kernel.linear_att_cpu_cache_copy",
        "copy_linear_att_state_to_linear_att_state",
    ),
    "common.basemodel.local_vocab_topk.local_vocab_topk": (
        "lightllm.common.basemodel.triton_kernel.local_vocab_topk",
        "local_vocab_topk",
    ),
    "common.basemodel.logprobs_capture.scatter_prompt_logprobs_to_cpu": (
        "lightllm.common.basemodel.triton_kernel.logprobs_capture",
        "scatter_prompt_logprobs_to_cpu",
    ),
    "common.basemodel.mhc.hc_contract": (
        "lightllm.common.basemodel.triton_kernel.mhc",
        "hc_contract",
    ),
    "common.basemodel.mhc.hc_expand": (
        "lightllm.common.basemodel.triton_kernel.mhc",
        "hc_expand",
    ),
    "common.basemodel.mhc.hc_post": (
        "lightllm.common.basemodel.triton_kernel.mhc",
        "hc_post",
    ),
    "common.basemodel.mhc.hc_pre_norm": (
        "lightllm.common.basemodel.triton_kernel.mhc",
        "hc_pre_norm",
    ),
    "common.basemodel.mtp_utils.gen_b_req_mtp_start_loc": (
        "lightllm.common.basemodel.triton_kernel.mtp_utils",
        "gen_b_req_mtp_start_loc",
    ),
    "common.basemodel.mtp_utils.linear_att_mtp_state_index_update": (
        "lightllm.common.basemodel.triton_kernel.mtp_utils",
        "linear_att_mtp_state_index_update",
    ),
    "common.basemodel.mtp_utils.mtp_scatter_next_token_ids": (
        "lightllm.common.basemodel.triton_kernel.mtp_utils",
        "mtp_scatter_next_token_ids",
    ),
    "common.basemodel.mtp_utils.mtp_verify": (
        "lightllm.common.basemodel.triton_kernel.mtp_utils",
        "mtp_verify",
    ),
    "common.basemodel.multimodal_emb.mark_multimodal_obj": (
        "lightllm.common.basemodel.triton_kernel.multimodal_emb",
        "mark_multimodal_obj",
    ),
    "common.basemodel.multimodal_emb.multimodal_emb": (
        "lightllm.common.basemodel.triton_kernel.multimodal_emb",
        "multimodal_emb",
    ),
    "common.basemodel.norm.gated_rmsnorm.gated_rmsnorm_forward": (
        "lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm",
        "gated_rmsnorm_forward",
    ),
    "common.basemodel.norm.layernorm.layernorm_forward": (
        "lightllm.common.basemodel.triton_kernel.norm.layernorm",
        "layernorm_forward",
    ),
    "common.basemodel.norm.qk_norm.qk_rmsnorm_forward": (
        "lightllm.common.basemodel.triton_kernel.norm.qk_norm",
        "qk_rmsnorm_forward",
    ),
    "common.basemodel.norm.qk_norm.qk_rmsnorm_fused_forward": (
        "lightllm.common.basemodel.triton_kernel.norm.qk_norm",
        "qk_rmsnorm_fused_forward",
    ),
    "common.basemodel.norm.rmsnorm.rmsnorm_forward": (
        "lightllm.common.basemodel.triton_kernel.norm.rmsnorm",
        "rmsnorm_forward",
    ),
    "common.basemodel.pack_vocab_parallel_topk.pack_vocab_parallel_topk": (
        "lightllm.common.basemodel.triton_kernel.pack_vocab_parallel_topk",
        "pack_vocab_parallel_topk",
    ),
    "common.basemodel.pack_vocab_parallel_topk.unpack_vocab_parallel_topk": (
        "lightllm.common.basemodel.triton_kernel.pack_vocab_parallel_topk",
        "unpack_vocab_parallel_topk",
    ),
    "common.basemodel.post_process.apply_constraint_mask.apply_constraint_mask": (
        "lightllm.common.basemodel.triton_kernel.post_process.apply_constraint_mask",
        "apply_constraint_mask",
    ),
    "common.basemodel.post_process.apply_invalid_token.apply_invalid_token_ids": (
        "lightllm.common.basemodel.triton_kernel.post_process.apply_invalid_token",
        "apply_invalid_token_ids",
    ),
    "common.basemodel.post_process.apply_penalty.apply_penalty": (
        "lightllm.common.basemodel.triton_kernel.post_process.apply_penalty",
        "apply_penalty",
    ),
    "common.basemodel.post_process.apply_penalty_gpu_cache.apply_penalty_gpu_cache": (
        "lightllm.common.basemodel.triton_kernel.post_process.apply_penalty_gpu_cache",
        "apply_penalty_gpu_cache",
    ),
    "common.basemodel.routing_capture.scatter_routing_topk_to_cpu": (
        "lightllm.common.basemodel.triton_kernel.routing_capture",
        "scatter_routing_topk_to_cpu",
    ),
    "common.basemodel.select_mtp_rows.select_accepted_tail_rows": (
        "lightllm.common.basemodel.triton_kernel.select_mtp_rows",
        "select_accepted_tail_rows",
    ),
    "common.basemodel.sp_pad_copy.sp_pad_copy": (
        "lightllm.common.basemodel.triton_kernel.sp_pad_copy",
        "sp_pad_copy",
    ),
    "models.deepseek2.rotary_emb.rotary_emb_fwd": (
        "lightllm.models.deepseek2.triton_kernel.rotary_emb",
        "rotary_emb_fwd",
    ),
    "models.deepseek2.sample_kv.sample_kv": (
        "lightllm.models.deepseek2.triton_kernel.sample_kv",
        "sample_kv",
    ),
    "models.deepseek2.weight_dequant.weight_dequant": (
        "lightllm.models.deepseek2.triton_kernel.weight_dequant",
        "weight_dequant",
    ),
    "models.deepseek3_2.act_quant.act_quant": (
        "lightllm.models.deepseek3_2.triton_kernel.act_quant",
        "act_quant",
    ),
    "models.deepseek3_2.destindex_copy_indexer_ks.destindex_copy_indexer_ks": (
        "lightllm.models.deepseek3_2.triton_kernel.destindex_copy_indexer_ks",
        "destindex_copy_indexer_ks",
    ),
    "models.deepseek3_2.destindex_copy_kv_flashmla_fp8.destindex_copy_kv_flashmla_fp8": (
        "lightllm.models.deepseek3_2.triton_kernel.destindex_copy_kv_flashmla_fp8",
        "destindex_copy_kv_flashmla_fp8",
    ),
    "models.deepseek3_2.extract_indexer_ks.extract_indexer_ks": (
        "lightllm.models.deepseek3_2.triton_kernel.extract_indexer_ks",
        "extract_indexer_ks",
    ),
    "models.deepseek3_2.hadamard_transform.hadamard_transform": (
        "lightllm.models.deepseek3_2.triton_kernel.hadamard_transform",
        "hadamard_transform",
    ),
    "models.deepseek3_2.prefill_compact_kv_flashmla_fp8.get_prefill_kv_cache_and_remap_indices_triton": (
        "lightllm.models.deepseek3_2.triton_kernel.prefill_compact_kv_flashmla_fp8",
        "get_prefill_kv_cache_and_remap_indices_triton",
    ),
    "models.deepseek3_2.topk_index_to_mem_index.trans_topk_index_to_mem_index": (
        "lightllm.models.deepseek3_2.triton_kernel.topk_index_to_mem_index",
        "trans_topk_index_to_mem_index",
    ),
    "models.gemma4.build_b_image_token_end.build_b_image_token_end": (
        "lightllm.models.gemma4.triton_kernel.build_b_image_token_end",
        "build_b_image_token_end",
    ),
    "models.gemma4.context_attention_fwd_gemma4_mm.context_attention_fwd_gemma4_mm": (
        "lightllm.models.gemma4.triton_kernel.context_attention_fwd_gemma4_mm",
        "context_attention_fwd_gemma4_mm",
    ),
    "models.gemma_2b.gelu_and_mul.gelu_and_mul_fwd": (
        "lightllm.models.gemma_2b.triton_kernel.gelu_and_mul",
        "gelu_and_mul_fwd",
    ),
    "models.glm5_next.index_quant.hadamard_transform_quant_fp8": (
        "lightllm.models.glm5_next.triton_kernel.index_quant",
        "hadamard_transform_quant_fp8",
    ),
    "models.glm5_next.kpool.compress_pools": (
        "lightllm.models.glm5_next.triton_kernel.kpool",
        "compress_pools",
    ),
    "models.glm5_next.kpool.expand_topk": (
        "lightllm.models.glm5_next.triton_kernel.kpool",
        "expand_topk",
    ),
    "models.glm5_next.kpool.gather_paged_pools": (
        "lightllm.models.glm5_next.triton_kernel.kpool",
        "gather_paged_pools",
    ),
    "models.glm5_next.kpool.gather_pools": (
        "lightllm.models.glm5_next.triton_kernel.kpool",
        "gather_pools",
    ),
    "models.glm5_next.kpool.get_pool_ranges": (
        "lightllm.models.glm5_next.triton_kernel.kpool",
        "get_pool_ranges",
    ),
    "models.llama.rotary_emb.rotary_emb_fwd": (
        "lightllm.models.llama.triton_kernel.rotary_emb",
        "rotary_emb_fwd",
    ),
    "models.llama.token_attention_nopad_att1.token_att_fwd": (
        "lightllm.models.llama.triton_kernel.token_attention_nopad_att1",
        "token_att_fwd",
    ),
    "models.neo_chat_moe.get_neo_position.get_neo_position_triton": (
        "lightllm.models.neo_chat_moe.triton_kernel.get_neo_position",
        "get_neo_position_triton",
    ),
    "models.phi3.rotary_emb.rotary_emb_fwd": (
        "lightllm.models.phi3.triton_kernel.rotary_emb",
        "rotary_emb_fwd",
    ),
    "models.qwen2_vl.get_mrope_position_ids.get_mrope_position_triton": (
        "lightllm.models.qwen2_vl.triton_kernel.get_mrope_position_ids",
        "get_mrope_position_triton",
    ),
    "models.qwen2_vl.mrope.mrope_triton_fused": (
        "lightllm.models.qwen2_vl.triton_kernel.mrope",
        "mrope_triton_fused",
    ),
    "models.qwen2_vl.rotary_pos_emb.apply_rotary_pos_emb_triton": (
        "lightllm.models.qwen2_vl.triton_kernel.rotary_pos_emb",
        "apply_rotary_pos_emb_triton",
    ),
    "models.qwen3_dflash2.selector_walk.greedy_selector_walk": (
        "lightllm.models.qwen3_dflash2.triton_kernel.selector_walk",
        "greedy_selector_walk",
    ),
    "models.qwen3_vl.deepstack_multimodal_emb.apply_deepstack_features": (
        "lightllm.models.qwen3_vl.triton_kernel.deepstack_multimodal_emb",
        "apply_deepstack_features",
    ),
    "models.qwen3next.shared_expert_gate.sigmoid_mul_": (
        "lightllm.models.qwen3next.triton_kernel.shared_expert_gate",
        "sigmoid_mul_",
    ),
    "models.vit.flashattention_nopad.flash_attention_fwd": (
        "lightllm.models.vit.triton_kernel.flashattention_nopad",
        "flash_attention_fwd",
    ),
    "models.vit.gelu_vit.gelu_fwd": (
        "lightllm.models.vit.triton_kernel.gelu_vit",
        "gelu_fwd",
    ),
    "models.vit.rms_norm_vit.qk_rms_norm": (
        "lightllm.models.vit.triton_kernel.rms_norm_vit",
        "qk_rms_norm",
    ),
    "models.vit.rms_norm_vit.rms_norm": (
        "lightllm.models.vit.triton_kernel.rms_norm_vit",
        "rms_norm",
    ),
}

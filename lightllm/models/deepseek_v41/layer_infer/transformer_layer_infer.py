import torch

from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer
from lightllm.common.basemodel.triton_kernel.mhc import hc_post, shifted_hc_pre_norm
from lightllm.common.quantization.deepgemm import DeepGEMMFP8w8a8B128QuantizationMethod
from lightllm.common.quantization.w8a8 import FP8w8a8B128TritonQuantizationMethod
from .indexer import DeepseekV41IndexInfer
from ..engram import apply_engram_to_streams
from ..infer_struct import DeepseekV41InferStateInfo
from ..layer_weights.transformer_layer_weight import DeepseekV41TransformerLayerWeight
from ..triton_kernel.attention import compress_and_normalize_kv_pairs, rotate_and_cache_compressed_kv
from ..triton_kernel.q_rope import apply_attention_query_rope


class DeepseekV41TransformerLayerInfer(DeepseekV4TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.is_kv_source = layer_num in network_config["kv_source_layer_ids"]
        self.engram_layer_index = (
            network_config["engram_layer_ids"].index(layer_num)
            if layer_num in network_config["engram_layer_ids"]
            else None
        )

    def _init_compress_and_index(self):
        self.index_infer = DeepseekV41IndexInfer(self.layer_num_, self.network_config_)

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: DeepseekV41InferStateInfo,
        layer_weight: DeepseekV41TransformerLayerWeight,
    ):
        qkv = layer_weight.wq_a_wkv_.mm(input)
        q_lora = layer_weight.q_norm_(qkv[:, : -self.head_dim_], eps=self.eps_)
        q = self._project_q(q_lora, infer_state, layer_weight)
        self._norm_rope_and_cache_kv(qkv[:, -self.head_dim_ :], infer_state, layer_weight)
        return q, q_lora, input

    def _norm_rope_and_cache_kv(self, kv, infer_state, layer_weight):
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_kv_norm_rope_to_swa

        state = infer_state.swa_state
        att_state = infer_state.prefill_att_state if infer_state.is_prefill else infer_state.decode_att_state
        swa, _ = att_state.backend.get_att_input_params(infer_state, self.layer_num_)
        fused_kv_norm_rope_to_swa(
            kv=kv,
            kv_norm_weight=layer_weight.kv_norm_.weight,
            eps=self.eps_,
            freqs_cis=self.freqs_cis,
            position_ids=infer_state.position_ids,
            swa_write_slots=infer_state.dsv4_swa_write_slots,
            swa_runtime_write_slots=infer_state.dsv4_swa_runtime_write_slots,
            swa_kv_buffer=swa,
            swa_page_token_num=state.config.page_token_size,
            req_to_swa=state.get_layer_view(logical_layer_id=self.layer_num_),
            bf16_rounding=True,
        )

    def _project_q(self, q_lora, infer_state, layer_weight):
        wq_b = layer_weight.wq_b_
        tokens = q_lora.shape[0]
        shape = (tokens, self.tp_q_head_num_, self.head_dim_)
        if self.tp_q_head_num_ == self.flashmla_q_head_num_:
            q = wq_b.mm(q_lora).view(shape)
            q_in = q_heads = q
        elif isinstance(
            wq_b.quant_method, (FP8w8a8B128TritonQuantizationMethod, DeepGEMMFP8w8a8B128QuantizationMethod)
        ):
            # These FP8 GEMMs can write into the valid heads with the padded token stride.
            q = self._alloc_q(tokens, infer_state, q_lora)
            q_in = q_heads = q[:, : self.tp_q_head_num_, :]
            wq_b.mm(q_lora, out=q_heads.flatten(1))
        else:
            q_in = wq_b.mm(q_lora).view(shape)
            q = self._alloc_q(tokens, infer_state, q_in)
            q_heads = q[:, : self.tp_q_head_num_, :]
        apply_attention_query_rope(q_in, q_heads, self.freqs_cis, infer_state.position_ids)
        return q

    def _hc_attn_in(self, input_embdings, infer_state, layer_weight):
        if isinstance(input_embdings, tuple):
            streams, pre_mix = input_embdings
        else:
            streams = input_embdings
            pre_mix = streams.new_zeros((streams.shape[0], self.hc_mult), dtype=torch.float32)
            pre_mix[:, 0] = 1
        if self.engram_layer_index is not None:
            streams = streams.view(-1, self.hc_mult, self.embed_dim_)
            streams = apply_engram_to_streams(
                streams,
                infer_state.engram_embedding_row_ids[:, self.engram_layer_index],
                layer_weight,
                self.eps_,
                infer_state.dist_group,
                text_token_mask=infer_state.input_ids < self.vocab_size if self.has_vision else None,
            ).flatten(1)
        x, res_mix, post_mix, attn_pre = shifted_hc_pre_norm(
            streams,
            layer_weight.hc_attn_fn_.weight,
            layer_weight.hc_attn_scale_.weight,
            layer_weight.hc_attn_base_.weight,
            layer_weight.attn_norm_.weight,
            self.hc_mult,
            self.eps_,
            self.eps_,
            self.hc_eps,
            self.sinkhorn_iters,
            previous_pre=pre_mix,
        )
        return x, streams, res_mix, post_mix, attn_pre

    def _hc_ffn_in(self, x, residual, res_mix, post_mix, attn_pre, layer_weight):
        streams = hc_post(x, residual, res_mix, post_mix, streams=self.hc_mult)
        x, res_mix, post_mix, ffn_pre = shifted_hc_pre_norm(
            streams,
            layer_weight.hc_ffn_fn_.weight,
            layer_weight.hc_ffn_scale_.weight,
            layer_weight.hc_ffn_base_.weight,
            layer_weight.ffn_norm_.weight,
            self.hc_mult,
            self.eps_,
            self.eps_,
            self.hc_eps,
            self.sinkhorn_iters,
            previous_pre=attn_pre,
        )
        return x, streams, res_mix, post_mix, ffn_pre

    def _hc_ffn_out(self, x, residual, res_mix, post_mix, ffn_pre):
        streams = hc_post(x, residual, res_mix, post_mix, streams=self.hc_mult)
        return streams, ffn_pre

    def context_forward(
        self, input_embdings, infer_state: DeepseekV41InferStateInfo, layer_weight: DeepseekV41TransformerLayerWeight
    ):
        x, residual, res_mix, post_mix, attn_pre = self._hc_attn_in(input_embdings, infer_state, layer_weight)
        x = self.context_attention_forward(x, infer_state, layer_weight)
        x, residual, res_mix, post_mix, ffn_pre = self._hc_ffn_in(
            x, residual, res_mix, post_mix, attn_pre, layer_weight
        )
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, res_mix, post_mix, ffn_pre)

    def token_forward(
        self, input_embdings, infer_state: DeepseekV41InferStateInfo, layer_weight: DeepseekV41TransformerLayerWeight
    ):
        x, residual, res_mix, post_mix, attn_pre = self._hc_attn_in(input_embdings, infer_state, layer_weight)
        x = self.token_attention_forward(x, infer_state, layer_weight)
        x, residual, res_mix, post_mix, ffn_pre = self._hc_ffn_in(
            x, residual, res_mix, post_mix, attn_pre, layer_weight
        )
        x = self._ffn(x, infer_state, layer_weight)
        return self._hc_ffn_out(x, residual, res_mix, post_mix, ffn_pre)

    def _compress_and_index(self, q_lora, x, infer_state: DeepseekV41InferStateInfo, layer_weight):
        if self.is_kv_source:
            self._write_global_kv(x, infer_state, layer_weight)
        return self.index_infer.build_metadata(x, q_lora, infer_state, layer_weight)

    def _write_global_kv(self, x, infer_state: DeepseekV41InferStateInfo, layer_weight):
        ratio = self.compress_ratio
        projected = self.alloc_tensor(
            (x.shape[0], self.head_dim_ * ratio), dtype=torch.float32 if ratio == 2 else x.dtype, device=x.device
        )
        layer_weight.compressor_wkv_gate_.mm(x, out=projected)
        if ratio == 2:
            kv_candidates, pooling_logits = projected.split(self.head_dim_, -1)
            latent = compress_and_normalize_kv_pairs(
                kv_candidates,
                pooling_logits,
                infer_state.dsv4_sparse_req_idx,
                infer_state.position_ids,
                infer_state.get_compressor_state(self.layer_num_).get_layer_view(self.layer_num_),
                infer_state.req_manager.HOLD_REQUEST_ID,
                layer_weight.compressor_norm_.weight,
                self.eps_,
            )
        else:
            latent = layer_weight.compressor_norm_(projected, eps=self.eps_)
        # Main KV and indexer K share the normalized latent, with separate projections and RoPE.
        self.index_infer.write_indexer_k(latent, infer_state, layer_weight)
        rotate_and_cache_compressed_kv(
            latent,
            self.freqs_cis,
            infer_state.mem_index,
            infer_state.position_ids,
            infer_state.get_compressed_kv_view(self.layer_num_).kv,
            ratio,
            infer_state.get_compressed_kv_state(self.layer_num_).config.block_token_size // ratio,
        )

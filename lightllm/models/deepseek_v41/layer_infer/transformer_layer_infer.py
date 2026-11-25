import torch

from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer
from lightllm.common.basemodel.triton_kernel.mhc import hc_post, hc_pre_norm
from .indexer import DeepseekV41IndexInfer
from ..engram import engram_forward
from ..infer_struct import DeepseekV41InferStateInfo
from ..layer_weights.transformer_layer_weight import DeepseekV41TransformerLayerWeight
from ..triton_kernel.attention import (
    pool_pairs,
    destindex_copy_kv_dsv41,
)


class DeepseekV41TransformerLayerInfer(DeepseekV4TransformerLayerInfer):
    q_head_norm = False
    kv_bf16_rounding = True

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.is_kv_source = layer_num in network_config["kv_source_layer_ids"]
        self.engram_index = (
            network_config["engram_layer_ids"].index(layer_num)
            if layer_num in network_config["engram_layer_ids"]
            else None
        )

    def _init_compress_and_index(self):
        self.index_infer = DeepseekV41IndexInfer(self.layer_num_, self.network_config_)

    def _hc_attn_in(self, input_embdings, infer_state, layer_weight):
        if isinstance(input_embdings, tuple):
            streams, pre_mix = input_embdings
        else:
            streams = input_embdings
            pre_mix = streams.new_zeros((streams.shape[0], self.hc_mult), dtype=torch.float32)
            pre_mix[:, 0] = 1
        if self.engram_index is not None:
            streams = streams.view(-1, self.hc_mult, self.embed_dim_)
            streams = engram_forward(
                streams,
                infer_state.engram_hashes[:, self.engram_index],
                layer_weight,
                self.eps_,
                infer_state.dist_group,
                token_mask=infer_state.input_ids < self.vocab_size if self.has_vision else None,
            ).flatten(1)
        x, res_mix, post_mix, attn_pre = hc_pre_norm(
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
        x, res_mix, post_mix, ffn_pre = hc_pre_norm(
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

    def _project_q(self, qa, infer_state, layer_weight):
        from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_q_norm_rope

        tokens = qa.shape[0]
        if self.tp_q_head_num_ == self.flashmla_q_head_num_:
            q = layer_weight.wq_b_.mm(qa).view(tokens, self.tp_q_head_num_, self.head_dim_)
            active_heads = q
        elif layer_weight.wq_b_.quant_method.method_name == "fp8w8a8-b128-deepgemm":
            # DeepGEMM accepts a row-strided BF16 output. Leave the workspace's padded heads untouched.
            q = self._alloc_q(tokens, infer_state, qa)
            active_heads = q[:, : self.tp_q_head_num_, :]
            layer_weight.wq_b_.mm(qa, out=active_heads.flatten(1))
        else:
            return super()._project_q(qa, infer_state, layer_weight)
        fused_q_norm_rope(
            active_heads, active_heads, self.eps_, self.freqs_cis, infer_state.position_ids, apply_norm=False
        )
        return q

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
            kv, score = projected.split(self.head_dim_, -1)
            latent = pool_pairs(
                kv,
                score,
                infer_state.dsv4_sparse_req_idx,
                infer_state.position_ids,
                infer_state.get_compressor_state(self.layer_num_).get_layer_view(self.layer_num_),
                infer_state.req_manager.HOLD_REQUEST_ID,
            )
        else:
            latent = projected
        latent = layer_weight.compressor_norm_(latent, eps=self.eps_)
        # Main KV and indexer K share the normalized latent, with separate projections and RoPE.
        self.index_infer.write_indexer_k(latent, infer_state, layer_weight)
        destindex_copy_kv_dsv41(
            latent,
            self.freqs_cis,
            infer_state.mem_index,
            infer_state.position_ids,
            infer_state.get_compressed_kv_view(self.layer_num_).kv,
            ratio,
            infer_state.get_compressed_kv_state(self.layer_num_).config.block_token_size // ratio,
        )

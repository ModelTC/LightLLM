from functools import partial
from typing import override

import torch
from sgl_kernel.flash_mla import flash_mla_sparse_fwd
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.deepseek3_2.layer_infer.nsa_indexer_layer_inder import NSAIndexerInfer
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2FlashAttentionInferStateInfo
from lightllm.models.deepseek3_2.triton_kernel.token_group_quant import per_token_group_quant_mla_deep_gemm_masked_fp8
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd


class Deepseek3_2TransformerLayerInfer(Deepseek2TransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        self.index_topk = network_config["index_topk"]
        super().__init__(layer_num, network_config, mode)

        self.indexer = NSAIndexerInfer(
            layer_idx=self.layer_num_,
            network_config=self.network_config_,
            mode=mode
        )
        return

    @override
    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Deepseek3_2FlashAttentionInferStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)

        if self.q_lora_rank is None:
            q = layer_weight.q_weight_.mm(input)
            cache_kv = layer_weight.kv_a_proj_with_mqa_.mm(input).view(-1, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        else:
            q, cache_kv = layer_weight.qkv_a_proj_with_mqa_.mm(input).split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_)

            self.indexer.hidden_states = input
            self.indexer.q_lora = q

            q = layer_weight.q_b_proj_.mm(q)
            cache_kv = cache_kv.view(-1, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        q = q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        rmsnorm_forward(
            cache_kv[:, :, : self.kv_lora_rank],
            weight=layer_weight.kv_a_layernorm_.weight,
            eps=self.eps_,
            out=cache_kv[:, :, : self.kv_lora_rank],
        )

        rotary_emb_fwd(
            q_rope,
            cache_kv[:, :, self.kv_lora_rank :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv

    @override
    def _bind_attention(self):
        self._context_attention_kernel = partial(Deepseek3_2TransformerLayerInfer._context_attention_flashmla_kernel_with_indexer, self)
        self._token_attention_kernel = partial(Deepseek3_2TransformerLayerInfer._token_attention_flashmla_kernel_with_indexer, self)
        pass

    def _context_attention_flashmla_kernel_with_indexer(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek3_2FlashAttentionInferStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        q_all = torch.cat([q_nope, q_rope], dim=-1)
        topk_indices = self.indexer.get_indices(
            infer_state,
            layer_weight.indexer_layer_weight,
        )
        mla_out, _, _ = flash_mla_sparse_fwd(
            q=q_all,
            kv=infer_state.mem_manager.kv_buffer[self.layer_num_],
            indices=topk_indices.unsqueeze(1),
            sm_scale=self.softmax_scale,
            d_v=self.kv_lora_rank,
        )
        return mla_out

    def _token_attention_flashmla_kernel_with_indexer(
        self, q, infer_state: Deepseek3_2FlashAttentionInferStateInfo, layer_weight: Deepseek3_2TransformerLayerWeight, out=None
    ):
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        k_rope = kv[:, :, -self.qk_rope_head_dim :].reshape(-1, 1, 1, self.qk_rope_head_dim)
        kv_nope = kv[:, :, : -self.qk_rope_head_dim].reshape(-1, 1, 1, self.kv_lora_rank)
        topk_indices = self.indexer.get_indices(
            infer_state,
            layer_weight.indexer_layer_weight,
        )
        o = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            page_table=topk_indices,
            cache_seqlens=infer_state.b_att_seq_len,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k_new=infer_state.cu_seqlens_k,
            max_seqlen_q=infer_state.max_q_seq_len,
            softmax_scale=self.softmax_scale,
            causal=True,
            softcap=0.0,
            return_softmax_lse=False,
            num_splits=0, # TODO enable_deterministic_inference
        )

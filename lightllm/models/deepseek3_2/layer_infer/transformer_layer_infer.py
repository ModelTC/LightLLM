from functools import partial
from typing import override
from venv import logger

import torch
from sgl_kernel.flash_mla import flash_mla_sparse_fwd
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.deepseek3_2.layer_infer.nsa_indexer_layer_inder import NSAIndexerInfer
from lightllm.models.deepseek3_2.layer_weights.transformer_layer_weight import Deepseek3_2TransformerLayerWeight
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2FlashAttentionStateInfo
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
        self.topk_indices = None
        return

    @override
    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Deepseek3_2FlashAttentionStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)

        q, cache_kv = layer_weight.qkv_a_proj_with_mqa_.mm(input).split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q = rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_)

        self.topk_indices = self.indexer.get_indices(input, q, infer_state, layer_weight.indexer_layer_weight)

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
        super()._bind_attention()
        self._context_attention_kernel = partial(Deepseek3_2TransformerLayerInfer._nsa_context_attention_kernel, self)
        self._token_attention_kernel = partial(Deepseek3_2TransformerLayerInfer._nsa_token_attention_kernel, self)
        pass

    def _nsa_context_attention_kernel(
        self,
        q: torch.Tensor,
        kv,
        infer_state: Deepseek3_2FlashAttentionStateInfo,
        layer_weight: Deepseek3_2TransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        q_all = torch.cat([q_nope, q_rope], dim=-1)

        mla_out, _, _ = flash_mla_sparse_fwd(
            q=q_all,
            kv=infer_state.mem_manager.kv_buffer[self.layer_num_],
            indices=self.topk_indices,
            sm_scale=self.softmax_scale,
            d_v=self.kv_lora_rank,
        )
        return mla_out

    def _nsa_token_attention_kernel(
        self, q, infer_state: Deepseek3_2FlashAttentionStateInfo, layer_weight: Deepseek3_2TransformerLayerWeight, out=None
    ):
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        k_rope = kv[:, :, -self.qk_rope_head_dim :].reshape(-1, 1, 1, self.qk_rope_head_dim)
        kv_nope = kv[:, :, : -self.qk_rope_head_dim].reshape(-1, 1, 1, self.kv_lora_rank)
        k_descale, v_descale = None, None
        o_tensor = flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_rope,
            v_cache=kv_nope,
            qv=q_nope,
            page_table=self.topk_indices,
            cache_seqlens=infer_state.b_att_seq_len,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k_new=infer_state.cu_seqlens_k,
            max_seqlen_q=infer_state.max_q_seq_len,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
        )
        return o_tensor
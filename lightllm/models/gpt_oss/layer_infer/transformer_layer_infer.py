import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from functools import partial
from typing import Optional

from lightllm.models.gpt_oss.layer_weights.transformer_layer_weight import GptOssTransformerLayerWeight
from lightllm.models.llama.flashattention_infer_struct import FlashAttentionStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.utils.sgl_utils import flash_attn_with_kvcache
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

class GptOssTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.hidden_size = self.network_config_['hidden_size']
        self.alpha = 1.702
        self.limit = 7.0
        self.top_k = network_config['num_experts_per_tok']
        self.sliding_window = network_config['sliding_window']
        self.head_dim_ = network_config["head_dim"]

    def _bind_attention(self):
        self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        self._context_attention_kernel = self._conext_sliding_attention_flashattention
        self._token_attention_kernel = self._token_sliding_attention_flashattention
    
    def _bind_norm(self):
        self._att_norm = self._att_norm
        self._ffn_norm = self._ffn_norm
        return

    def _experts(self, hidden_states: torch.Tensor, router_indices, routing_weights, layer_weight: GptOssTransformerLayerWeight):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]

        hidden_states = hidden_states.repeat(num_experts, 1)
        hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(hidden_states, layer_weight.gate_up_proj_weight) + layer_weight.gate_up_proj_bias.weight[..., None, :]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = torch.bmm(((up + 1) * glu), layer_weight.down_proj_weight)
        next_states = next_states + layer_weight.down_proj_bias.weight[..., None, :]
        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)
        return next_states
    
    def _att_norm(
        self, input, infer_state, layer_weight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self._gpt_oss_rmsnorm(input, weight=layer_weight.att_norm_weight_.weight, eps=self.eps_)
        return out
    
    def _ffn_norm(
        self, input, infer_state, layer_weight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self._gpt_oss_rmsnorm(input, weight=layer_weight.ffn_norm_weight_.weight, eps=self.eps_)
        return out
    
    def _gpt_oss_rmsnorm(self, hidden_states, weight, eps=1e-6):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return (weight * hidden_states).to(input_dtype)  # main diff with Llama

    def _router(self, hidden_states, layer_weight: GptOssTransformerLayerWeight):
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = layer_weight.moe_gate.mm(hidden_states)  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices
    
    def _ffn(self, input, infer_state: FlashAttentionStateInfo, layer_weight: GptOssTransformerLayerWeight) -> torch.Tensor:
        router_scores, router_indices = self._router(input, layer_weight)  # (num_experts, seq_len)
        routed_out = self._experts(input, router_indices=router_indices, routing_weights=router_scores, layer_weight=layer_weight)
        return routed_out
    
    def _conext_sliding_attention_flashattention(self, q, kv, infer_state: FlashAttentionStateInfo, layer_weight, out=None):
        if self.network_config_['layer_types'][self.layer_num_] == "sliding_attention":
            window_size = (self.sliding_window-1, self.sliding_window-1)
        else:
            window_size = (-1, -1)
        
        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :].reshape(
            -1, 1, self.tp_k_head_num_, self.head_dim_
        )
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ].reshape(-1, 1, self.tp_v_head_num_, self.head_dim_)
        q = q.reshape(-1, self.tp_q_head_num_, self.head_dim_)
        k_descale, v_descale = None, None  # disable quantization
        Lq = q.shape[-1]
        sm_scale = 1.0 / (Lq ** 0.5)
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=infer_state.page_table,
            cache_seqlens=infer_state.b_seq_len,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k_new=infer_state.cu_seqlens_k,
            max_seqlen_q=infer_state.q_max_seq_len,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            sinks=layer_weight.attn_sinks.weight,
        )
        return o

    def _token_sliding_attention_flashattention(self, q, infer_state: FlashAttentionStateInfo, layer_weight, out=None):
        if self.network_config_['layer_types'][self.layer_num_] == "sliding_attention":
            window_size = (self.sliding_window-1, self.sliding_window-1)
        else:
            window_size = (-1, -1)
        
        cache_k = infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :].reshape(
            -1, 1, self.tp_k_head_num_, self.head_dim_
        )
        cache_v = infer_state.mem_manager.kv_buffer[self.layer_num_][
            :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
        ].reshape(-1, 1, self.tp_v_head_num_, self.head_dim_)
        q = q.reshape(-1, self.tp_q_head_num_, self.head_dim_)
        k_descale, v_descale = None, None  # disable quantization
        Lq = q.shape[-1]
        sm_scale = 1.0 / (Lq ** 0.5)
        o = flash_attn_with_kvcache(
            q=q,
            k_cache=cache_k,
            v_cache=cache_v,
            page_table=infer_state.page_table,
            cache_seqlens=infer_state.b_seq_len,
            cu_seqlens_q=infer_state.cu_seqlens_q,
            cu_seqlens_k_new=infer_state.cu_seqlens_k,
            max_seqlen_q=1,
            softmax_scale=sm_scale,
            causal=True,
            window_size=window_size,
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            sinks=layer_weight.attn_sinks.weight,
        )
        return o
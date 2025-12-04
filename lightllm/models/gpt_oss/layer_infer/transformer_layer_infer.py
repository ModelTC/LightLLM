import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from functools import partial
from typing import Optional

from lightllm.models.gpt_oss.layer_weights.transformer_layer_weight import GptOssTransformerLayerWeight
from lightllm.models.llama.flashattention_infer_struct import FlashAttentionStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.utils.sgl_utils import flash_attn_with_kvcache
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class GptOssTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.hidden_size = self.network_config_["hidden_size"]
        self.alpha = 1.702
        self.limit = 7.0
        self.top_k = network_config["num_experts_per_tok"]
        self.sliding_window = network_config["sliding_window"]
        self.head_dim_ = network_config["head_dim"]

    def _bind_attention(self):
        self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        self._context_attention_kernel = self._context_sliding_attention_flashattention
        self._token_attention_kernel = self._token_sliding_attention_flashattention

    def _bind_norm(self):
        self._att_norm = self._att_norm
        self._ffn_norm = self._ffn_norm
        return

    def _att_norm(self, input, infer_state, layer_weight) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self._gpt_oss_rmsnorm(input, weight=layer_weight.att_norm_weight_.weight, eps=self.eps_)
        return out

    def _ffn_norm(self, input, infer_state, layer_weight) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self._gpt_oss_rmsnorm(input, weight=layer_weight.ffn_norm_weight_.weight, eps=self.eps_)
        return out

    def _gpt_oss_rmsnorm(self, hidden_states, weight, eps=1e-6):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return (weight * hidden_states).to(input_dtype)  # main diff with Llama

    def _ffn(
        self, input, infer_state: FlashAttentionStateInfo, layer_weight: GptOssTransformerLayerWeight
    ) -> torch.Tensor:
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape
        router_logits = layer_weight.moe_gate.mm(hidden_states)
        hidden_states = layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=True,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
        )
        return hidden_states.view(num_tokens, hidden_dim)

    def _context_sliding_attention_flashattention(
        self, q, kv, infer_state: FlashAttentionStateInfo, layer_weight, out=None
    ):
        if self.network_config_["layer_types"][self.layer_num_] == "sliding_attention":
            window_size = (self.sliding_window - 1, self.sliding_window - 1)
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
            window_size=window_size,
            softcap=0.0,
            k_descale=k_descale,
            v_descale=v_descale,
            return_softmax_lse=False,
            sinks=layer_weight.attn_sinks.weight,
        )
        return o

    def _token_sliding_attention_flashattention(self, q, infer_state: FlashAttentionStateInfo, layer_weight, out=None):
        if self.network_config_["layer_types"][self.layer_num_] == "sliding_attention":
            window_size = (self.sliding_window - 1, self.sliding_window - 1)
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

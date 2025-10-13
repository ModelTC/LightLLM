import time
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.models.bloom.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.bloom.triton_kernel.token_flashattention_nopad import token_attention_fwd
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward
from lightllm.common.basemodel import InferStateInfo
from lightllm.utils.infer_utils import mark_cost_time


class BloomTransformerLayerInfer(TransformerLayerInferTpl):
    """ """

    def __init__(self, layer_num, network_config, mode):
        super().__init__(layer_num, network_config, mode)
        self.eps_ = network_config["layer_norm_epsilon"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = self.tp_q_head_num_
        self.tp_v_head_num_ = self.tp_q_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["n_embed"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["n_embed"]
        return

    def _att_norm(self, input, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight) -> torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.att_norm_weight_.weight,
            bias=layer_weight.att_norm_weight_.bias,
            eps=self.eps_,
        )

    def _ffn_norm(self, input, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight) -> torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.ffn_norm_weight_.weight,
            bias=layer_weight.ffn_norm_weight_.bias,
            eps=self.eps_,
        )

    def _get_qkv(
        self, input, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = layer_weight.q_proj.mm(input.view(-1, self.embed_dim_))
        cache_kv = self._pre_cache_kv(infer_state=infer_state, layer_weight=layer_weight)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        return q, cache_kv

    def _context_attention_kernel(
        self, q, kv, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        context_attention_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            kv[:, 0 : self.tp_k_head_num_, :],
            kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
            infer_state.b_req_idx,
            layer_weight.tp_alibi,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.b_ready_cache_len,
            infer_state.max_len_in_batch,
            infer_state.req_manager.req_to_token_indexs,
        )
        return o_tensor

    def _token_attention_kernel(
        self, q, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight, out=None
    ) -> torch.Tensor:
        o_tensor = self.alloc_tensor(q.shape, q.dtype) if out is None else out
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        token_attention_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            kv[:, 0 : self.tp_k_head_num_, :],
            kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
            layer_weight.tp_alibi,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
            infer_state.total_token_num,
            alloc_tensor_func=self.alloc_tensor,
        )
        return o_tensor

    def _get_o(self, input, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight) -> torch.Tensor:
        o_tensor = layer_weight.o_proj.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_))
        return o_tensor

    def _ffn(self, input, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight) -> torch.Tensor:
        ffn1_out = layer_weight.gate_up_proj.mm(input.view(-1, self.embed_dim_))
        input = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate="tanh")
        ffn1_out = None
        ffn2_out = layer_weight.down_proj.mm(gelu_out)
        gelu_out = None
        return ffn2_out

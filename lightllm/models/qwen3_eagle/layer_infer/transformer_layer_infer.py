import os
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from typing import Tuple
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.qwen3_eagle.layer_weights.transformer_layer_weight import Qwen3EagleTransformerLayerWeight
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3EagleTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.hidden_size = network_config["hidden_size"]
        self.head_dim_ = network_config["head_dim"]
        return

    def _hidden_norm(
        self, input, infer_state: InferStateInfo, 
        layer_weight: Qwen3EagleTransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.hidden_norm_weight_._native_forward(
            input=input, eps=self.eps_, alloc_func=self.alloc_tensor)

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        tgt_hidden = infer_state.mtp_draft_input_hiddens
        tgt_hidden1 = self._hidden_norm(tgt_hidden, infer_state, layer_weight)
        
        input0 = self._att_norm(input_embdings, infer_state, layer_weight)
        input1 = torch.cat([input0, tgt_hidden1], dim=-1)
        input0 = None
        
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight) 

        o = self._context_attention_wrapper_run(
            q=q, cache_kv=cache_kv, infer_state=infer_state, layer_weight=layer_weight
        )

        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        tgt_hidden.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(tgt_hidden, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        tgt_hidden.add_(ffn_out.view(-1, self.embed_dim_))
        return tgt_hidden
        
        
    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        tgt_hidden = infer_state.mtp_draft_input_hiddens 
        tgt_hidden1 = self._hidden_norm(tgt_hidden, infer_state, layer_weight)
        
        input0 = self._att_norm(input_embdings, infer_state, layer_weight)
        input1 = torch.cat([input0, tgt_hidden1], dim=-1)
        input0 = None
        
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        tgt_hidden.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(tgt_hidden, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        tgt_hidden.add_(ffn_out.view(-1, self.embed_dim_))
        return tgt_hidden

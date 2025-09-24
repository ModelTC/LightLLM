import torch
import torch.functional as F
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from typing import Tuple
from functools import partial
import triton

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.distributed import all_reduce
from lightllm.models.cohere.layer_infer.transformer_layer_infer import CohereTransformerLayerInfer
from lightllm.models.cohere.triton_kernels.layernorm import layernorm_forward
from lightllm.models.gemma3.layer_weights.transformer_layer_weight import Gemma3TransformerLayerWeight
from lightllm.models.gemma_2b.triton_kernel.gelu_and_mul import gelu_and_mul_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd


class Gemma3TransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.tp_k_head_num_ = network_config["num_key_value_heads"]
        self.tp_v_head_num_ = network_config["num_key_value_heads"]
        self.eps_ = 1e-6
        self.head_dim_ = 256
        self.sliding_window_pattern = 6
        return

    def gemma3_rmsnorm(self, input, weight, eps: float = 1e-6, out=None):
        def _norm(x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(input.float())
        output = output * (1.0 + weight.float())
        if out is not None:
            out = output.to(out.dtype)
        return output

    def _pre_feedforward_layernorm(self, input, infer_state, layer_weight: Gemma3TransformerLayerWeight):
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self.gemma3_rmsnorm(input, layer_weight.pre_feedforward_layernorm_weight_.weight, self.eps_, out=out)
        return out

    def _post_feedforward_layernorm(self, input, infer_state, layer_weight: Gemma3TransformerLayerWeight):
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self.gemma3_rmsnorm(input, layer_weight.post_feedforward_layernorm_weight_.weight, self.eps_, out=out)
        return out

    def _k_norm(self, input, infer_state, layer_weight: Gemma3TransformerLayerWeight):
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self.gemma3_rmsnorm(input, layer_weight.k_norm_weight_.weight, self.eps_, out=out)
        return out

    def _q_norm(self, input, infer_state, layer_weight: Gemma3TransformerLayerWeight):
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self.gemma3_rmsnorm(input, layer_weight.q_norm_weight_.weight, self.eps_, out=out)
        return out

    def _att_norm(self, input, infer_state, layer_weight):
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self.gemma3_rmsnorm(input, layer_weight.att_norm_weight_.weight, self.eps_, out=out)
        return out

    def _ffn_norm(self, input, infer_state, layer_weight):
        out = self.alloc_tensor(input.shape, input.dtype)
        out = self.gemma3_rmsnorm(input, layer_weight.ffn_norm_weight_.weight, self.eps_, out=out)
        return out

    def _bind_norm(self):
        self._att_norm = partial(Gemma3TransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(Gemma3TransformerLayerInfer._ffn_norm, self)
        self._q_norm = partial(Gemma3TransformerLayerInfer._q_norm, self)
        self._k_norm = partial(Gemma3TransformerLayerInfer._k_norm, self)
        self._pre_feedforward_layernorm = partial(Gemma3TransformerLayerInfer._pre_feedforward_layernorm, self)
        self._post_feedforward_layernorm = partial(Gemma3TransformerLayerInfer._post_feedforward_layernorm, self)

    def _get_qkv(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma3TransformerLayerWeight
    ) -> torch.Tensor:
        q = layer_weight.q_proj.mm(input)
        # kv = layer_weight.kv_proj.mm(input)
        # kv = kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        k = layer_weight.k_proj.mm(input)
        v = layer_weight.v_proj.mm(input)
        cache_kv = self._pre_cache_kv(infer_state=infer_state, layer_weight=layer_weight)
        cache_kv[:, 0 : self.tp_k_head_num_, :] = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        cache_kv[:, self.tp_k_head_num_ :, :] = v.view(-1, self.tp_v_head_num_, self.head_dim_)

        # gemma3 use qk norm
        q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        k = cache_kv[:, 0 : self.tp_k_head_num_, :]
        q = self._q_norm(q.float(), infer_state, layer_weight).to(cache_kv.dtype)
        cache_kv[:, 0 : self.tp_k_head_num_, :] = self._k_norm(k.float(), infer_state, layer_weight).to(cache_kv.dtype)

        is_sliding = bool((self.layer_num_ + 1) % self.sliding_window_pattern)
        if is_sliding:
            rotary_emb_fwd(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                cache_kv[:, 0 : self.tp_k_head_num_, :],
                infer_state.position_cos_local.to(q.dtype),
                infer_state.position_sin_local.to(q.dtype),
            )
        else:
            rotary_emb_fwd(
                q.view(-1, self.tp_q_head_num_, self.head_dim_),
                cache_kv[:, 0 : self.tp_k_head_num_, :],
                infer_state.position_cos_global.to(q.dtype),
                infer_state.position_sin_global.to(q.dtype),
            )
        return q, cache_kv

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma3TransformerLayerWeight) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        gate = layer_weight.gate_proj.mm(input.view(-1, self.embed_dim_))
        up = layer_weight.up_proj.mm(input.view(-1, self.embed_dim_))
        # gelu_and_mul_fwd(up_gate_out, ffn1_out)
        ffn1_out = nn.functional.gelu(gate, approximate="tanh") * up
        input = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        ffn1_out = None
        return ffn2_out

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input_embdings = input_embdings.to(torch.bfloat16)
        input1 = self._att_norm(input_embdings.view(-1, self.embed_dim_).float(), infer_state, layer_weight).to(
            torch.bfloat16
        )
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        o = self._ffn_norm(o.float(), infer_state, layer_weight).to(torch.bfloat16)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._pre_feedforward_layernorm(input_embdings.float(), infer_state, layer_weight).to(torch.bfloat16)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        ffn_out = self._post_feedforward_layernorm(ffn_out.float(), infer_state, layer_weight).to(torch.bfloat16)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input_embdings = input_embdings.to(torch.bfloat16)
        input1 = self._att_norm(input_embdings.view(-1, self.embed_dim_).float(), infer_state, layer_weight).to(
            torch.bfloat16
        )
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        o = self._ffn_norm(o.float(), infer_state, layer_weight).to(torch.bfloat16)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._pre_feedforward_layernorm(input_embdings.float(), infer_state, layer_weight).to(torch.bfloat16)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        ffn_out = self._post_feedforward_layernorm(ffn_out.float(), infer_state, layer_weight).to(torch.bfloat16)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial
import triton

from lightllm.models.vit.layer_weights.transformer_layer_weight import ViTTransformerLayerWeight
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward, torch_rms_norm
from lightllm.models.vit.triton_kernel.flashattention_nopad import flash_attention_fwd


class ViTTransformerLayerInfer:
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        self.eps_ = network_config["layer_norm_eps"]
        self.head_num = network_config["num_attention_heads"]
        self.tp_padding_head_num = network_config["padding_head_num"] // world_size
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self.qk_norm = network_config["qk_normalization"]
        self.tp_padding_embed_dim_ = self.tp_padding_head_num * self.head_dim_
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.network_config_ = network_config
        self.mode = mode
        self.layer_num_ = layer_num
        return

    def norm(self, input, weight):
        input_dtype = input.dtype
        input_shape = input.shape
        input = input.view(-1, self.tp_padding_head_num * self.head_dim_)
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps_)
        out = weight * input.to(input_dtype)
        out = out.reshape(input_shape)
        return out

    def tp_norm(self, input, weight):
        input_shape = input.shape
        input = input.view(-1, self.tp_padding_head_num * self.head_dim_)
        input_dtype = input.dtype
        input = input.to(torch.float32)
        tp_variance = input.pow(2).sum(-1, keepdim=True)
        if self.world_size_ > 1:
            dist.all_reduce(tp_variance, op=dist.ReduceOp.SUM, async_op=False)
        variance = tp_variance / self.embed_dim_
        input = input * torch.rsqrt(variance + self.eps_)
        out = weight * input.to(input_dtype)
        out = out.reshape(input_shape)
        return out

    def _att_norm(self, input, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        if layer_weight.norm_type == "rms_norm":
            b = rmsnorm_forward(input, weight=layer_weight.att_norm_weight_.weight, eps=self.eps_)
        else:
            b = torch.nn.functional.layer_norm(
                input,
                normalized_shape=[1024],
                weight=layer_weight.att_norm_weight_.weight,
                bias=layer_weight.att_norm_weight_.bias,
                eps=layer_weight.layer_norm_eps,
            )
        return b

    def _ffn_norm(self, input, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        if layer_weight.norm_type == "rms_norm":
            return rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_.weight, eps=self.eps_)
        else:
            return torch.nn.functional.layer_norm(
                input,
                normalized_shape=[1024],
                weight=layer_weight.ffn_norm_weight_.weight,
                bias=layer_weight.ffn_norm_weight_.bias,
                eps=layer_weight.layer_norm_eps,
            )

    def _qk_norm(self, q, k, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        q_norm = self.tp_norm(q, layer_weight.q_norm_weight_.weight)
        k_norm = self.tp_norm(k, layer_weight.k_norm_weight_.weight)
        return q_norm, k_norm

    def _get_qkv(self, input, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        qkv = layer_weight.qkv_proj.mm(input.view(-1, self.embed_dim_), use_custom_tensor_mananger=False)
        qkv = qkv.view(batch_size, seq_len, 3, -1, self.head_dim_)
        q, k, v = qkv.unbind(2)
        return q, k, v

    def _context_attention_kernel(self, q, k, v) -> torch.Tensor:
        out = torch.empty_like(q)
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        flash_attention_fwd(q, k, v, out)
        return out.reshape(batch_size, seq_len, -1)

    def _get_o(self, input, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        o_tensor = layer_weight.o_proj.mm(
            input.view(-1, self.tp_padding_head_num * self.head_dim_), use_custom_tensor_mananger=False
        )
        if layer_weight.use_ls:
            o_tensor *= layer_weight.ls1
        return o_tensor.reshape((batch_size, seq_len, -1))

    def _ffn(self, input, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        fc1 = layer_weight.ffn_1_proj_.mm(input.view(-1, self.embed_dim_), use_custom_tensor_mananger=False)
        ffn1_out = torch.nn.functional.gelu(fc1)
        input_shape = input.shape
        input = None
        ffn2_out = layer_weight.ffn_2_proj_.mm(ffn1_out, use_custom_tensor_mananger=False)
        if layer_weight.use_ls:
            ffn2_out *= layer_weight.ls2
        ffn1_out = None
        return ffn2_out.reshape(input_shape)

    def _context_attention(self, input_embding, layer_weight):
        input1 = self._att_norm(input_embding, layer_weight)
        q, k, v = self._get_qkv(input1, layer_weight)
        if layer_weight.qk_norm:
            q, k = self._qk_norm(q, k, layer_weight)
        o = self._context_attention_kernel(q, k, v)
        o = self._get_o(o, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o)
        return

    def _context_ffn(self, input_embdings, layer_weight):
        input1 = self._ffn_norm(input_embdings, layer_weight)
        ffn_out = self._ffn(input1, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out)
        return

    def forward(self, input_embdings, layer_weight):
        self._context_attention(input_embdings, layer_weight=layer_weight)
        self._context_ffn(input_embdings, layer_weight)
        return input_embdings

import torch
import torch.distributed as dist

from typing import Tuple
from lightllm.models.vit.layer_weights.transformer_layer_weight import ViTTransformerLayerWeight
from lightllm.models.vit.triton_kernel.flashattention_nopad import flash_attention_fwd
from lightllm.utils.dist_utils import get_current_rank_in_dp, get_dp_world_size
from lightllm.models.vit.triton_kernel.gelu_vit import gelu_fwd
from lightllm.models.vit.triton_kernel.rms_norm_vit import rms_norm
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


class ViTTransformerLayerInfer:
    """ """

    def __init__(self, layer_num, network_config, mode=[]):
        self.tp_rank_ = get_current_rank_in_dp()
        self.tp_world_size_ = get_dp_world_size()
        self.eps_ = network_config["layer_norm_eps"]
        self.head_num = network_config["num_attention_heads"]
        self.tp_padding_head_num = network_config["padding_head_num"] // self.tp_world_size_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self.qk_norm = network_config["qk_normalization"]
        self.tp_padding_embed_dim_ = self.tp_padding_head_num * self.head_dim_

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
        if self.tp_world_size_ > 1:
            dist.all_reduce(tp_variance, op=dist.ReduceOp.SUM, async_op=False)
        variance = tp_variance / self.embed_dim_
        input = input * torch.rsqrt(variance + self.eps_)
        out = weight * input.to(input_dtype)
        out = out.reshape(input_shape)
        return out

    def _att_norm(self, input, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        if layer_weight.norm_type == "rms_norm":
            b = rms_norm(
                input, weight=layer_weight.att_norm_weight_.weight, eps=self.eps_, use_custom_tensor_mananger=True
            )
        else:
            b = torch.nn.functional.layer_norm(
                input,
                normalized_shape=[input.shape[-1]],
                weight=layer_weight.att_norm_weight_.weight,
                bias=layer_weight.att_norm_weight_.bias,
                eps=layer_weight.layer_norm_eps,
            )
        return b

    def _ffn_norm(self, input, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        if layer_weight.norm_type == "rms_norm":
            return rms_norm(
                input, weight=layer_weight.ffn_norm_weight_.weight, eps=self.eps_, use_custom_tensor_mananger=True
            )
        else:
            return torch.nn.functional.layer_norm(
                input,
                normalized_shape=[input.shape[-1]],
                weight=layer_weight.ffn_norm_weight_.weight,
                bias=layer_weight.ffn_norm_weight_.bias,
                eps=layer_weight.layer_norm_eps,
            )

    def _qk_norm(self, q, k, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        q_norm = self.tp_norm(q, layer_weight.q_norm_weight_.weight)
        k_norm = self.tp_norm(k, layer_weight.k_norm_weight_.weight)
        return q_norm, k_norm

    def _get_qkv(
        self, input, layer_weight: ViTTransformerLayerWeight
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        qkv = layer_weight.qkv_proj.mm(input.view(-1, self.embed_dim_), use_custom_tensor_mananger=True)
        qkv = qkv.view(batch_size, seq_len, 3, -1, self.head_dim_)
        q, k, v = qkv.unbind(2)
        return q, k, v

    def _context_attention_kernel(self, q, k, v) -> torch.Tensor:
        out = g_cache_manager.alloc_tensor(q.shape, q.dtype, device=q.device)
        batch_size, seq_len, head_num, head_dim = q.shape
        total_len = batch_size * seq_len
        reshape = lambda t: t.view(total_len, head_num, head_dim)
        q, k, v, out = map(reshape, (q, k, v, out))
        cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device) * seq_len
        max_seqlen = seq_len
        flash_attention_fwd(q, k, v, out, cu_seqlens, max_seqlen)
        return out.reshape(batch_size, seq_len, -1)

    def _get_o(self, input, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        o_tensor = layer_weight.o_proj.mm(
            input.view(-1, self.tp_padding_head_num * self.head_dim_), use_custom_tensor_mananger=True
        )
        if layer_weight.use_ls:
            o_tensor.mul_(layer_weight.ls1)
        return o_tensor.reshape((batch_size, seq_len, -1))

    def _ffn(self, input, layer_weight: ViTTransformerLayerWeight) -> torch.Tensor:
        fc1 = layer_weight.ffn_1_proj_.mm(input.view(-1, self.embed_dim_), use_custom_tensor_mananger=True)
        input_shape = input.shape
        input = None
        ffn1_out = gelu_fwd(fc1, use_custom_tensor_mananger=True)
        ffn2_out = layer_weight.ffn_2_proj_.mm(ffn1_out, use_custom_tensor_mananger=True)
        ffn1_out = None
        if layer_weight.use_ls:
            ffn2_out.mul_(layer_weight.ls2)
        return ffn2_out.reshape(input_shape)

    def _context_attention(self, input_embding, layer_weight):
        input1 = self._att_norm(input_embding, layer_weight)
        q, k, v = self._get_qkv(input1, layer_weight)
        input1 = None
        if layer_weight.qk_norm:
            q, k = self._qk_norm(q, k, layer_weight)
        o = self._context_attention_kernel(q, k, v)
        q = None
        k = None
        v = None
        o = self._get_o(o, layer_weight)
        if self.tp_world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o)
        return

    def _context_ffn(self, input_embdings, layer_weight):
        input1 = self._ffn_norm(input_embdings, layer_weight)
        ffn_out = self._ffn(input1, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out)
        return

    def forward(self, input_embdings, layer_weight):
        self._context_attention(input_embdings, layer_weight=layer_weight)
        self._context_ffn(input_embdings, layer_weight)
        return input_embdings

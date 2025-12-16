import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Tuple
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton
from lightllm.models.qwen3.layer_infer.transformer_layer_infer import Qwen3TransformerLayerInfer
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from lightllm.distributed import all_reduce
from lightllm.utils.dist_utils import get_global_world_size
from lightllm.models.qwen3_vl.triton_kernel.deepstack_multimodal_emb import apply_deepstack_features
from lightllm.models.qwen2_vl.layer_infer.transformer_layer_infer import Qwen2VLTransformerLayerInfer


class Qwen3VLTransformerLayerInfer(Qwen2VLTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.mrope_section = network_config["rope_scaling"]["mrope_section"]
        mrope_length = sum(self.mrope_section)
        self.axis_map = torch.zeros((1, mrope_length), dtype=torch.int32, device="cuda")
        for dim in (1, 2):
            length = self.mrope_section[dim] * 3
            self.axis_map[..., dim:length:3] = dim

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3TransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len, _ = input.shape
        input = input.view(-1, self.embed_dim_)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        rmsnorm_forward(
            q.view(-1, self.head_dim_),
            weight=layer_weight.q_norm_weight_.weight,
            eps=self.eps_,
            out=q.view(-1, self.head_dim_),
        )

        cache_kv[:, : self.tp_k_head_num_, :] = rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_, :].reshape(-1, cache_kv.shape[-1]),
            weight=layer_weight.k_norm_weight_.weight,
            eps=self.eps_,
        ).view(-1, self.tp_k_head_num_, cache_kv.shape[-1])

        q = q.view(1, seq_len, -1, self.head_dim_).transpose(1, 2)
        self.axis_map = self.axis_map.to(q.device)
        k = cache_kv[:, : self.tp_k_head_num_, :].view(1, seq_len, -1, self.head_dim_).transpose(1, 2)

        new_q, new_k = mrope_triton(
            q,
            k,
            infer_state.position_cos,
            infer_state.position_sin,
            self.axis_map,
        )
        new_q = new_q.transpose(1, 2).reshape(1, seq_len, -1)
        cache_kv[:, : self.tp_k_head_num_, :] = new_k.squeeze(0).permute(1, 0, 2)
        return new_q, cache_kv

    def context_forward(self, input_embdings, infer_state: Qwen3VLInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        apply_deepstack_features(
            input_embeddings=input_embdings,
            infer_state=infer_state,
            layer_num=self.layer_num_,
        )
        return input_embdings

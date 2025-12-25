import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial

from lightllm.distributed import all_reduce
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.glm4v.layer_weight.transformer_layer_weight import Glm4VTransformerLayerWeight


class Glm4VTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        mrope_section = network_config["rope_parameters"]["mrope_section"]
        self.mrope_section = torch.tensor(mrope_section, dtype=torch.int32, device="cuda")
        self.partial_rotary_factor = network_config["rope_parameters"]["partial_rotary_factor"]

    def _post_self_att_norm(
        self, input, infer_state: Qwen2VLInferStateInfo, layer_weight: Glm4VTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        rmsnorm_forward(input, weight=layer_weight._post_self_att_norm_weight_.weight, eps=self.eps_, out=out)
        return out

    def _post_mlp_norm(
        self, input, infer_state: Qwen2VLInferStateInfo, layer_weight: Glm4VTransformerLayerWeight
    ) -> torch.Tensor:
        out = self.alloc_tensor(input.shape, input.dtype)
        rmsnorm_forward(input, weight=layer_weight._post_mlp_norm_weight_.weight, eps=self.eps_, out=out)
        return out

    def _get_qkv(self, input, infer_state, layer_weight):
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        mrope_triton_fused(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            self.mrope_section,
            partial_rotary_factor=self.partial_rotary_factor,
            is_interleaved=False,
            is_glm4v=True,
        )
        return q, cache_kv

    def context_forward(self, input_embdings, infer_state: Qwen2VLInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)

        o = self._TransformerLayerInferTpl__context_attention_wrapper_run(
            q=q, cache_kv=cache_kv, infer_state=infer_state, layer_weight=layer_weight
        )

        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        o = self._post_self_att_norm(o, infer_state, layer_weight)  # add前多一次norm
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        ffn_out = self._post_mlp_norm(ffn_out, infer_state, layer_weight)  # mlp之后多一次norm
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(self, input_embdings, infer_state: Qwen2VLInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        o = self._post_self_att_norm(o, infer_state, layer_weight)  # add前多一次norm
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        ffn_out = self._post_mlp_norm(ffn_out, infer_state, layer_weight)  # mlp之后多一次norm
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=dist.ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def _tpsp_get_qkv(self, input, infer_state, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO
        raise Exception("not impl")

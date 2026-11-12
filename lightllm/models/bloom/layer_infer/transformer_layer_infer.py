from typing import Tuple

import torch

from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.basemodel import InferStateInfo, TransformerLayerInferTpl
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight


class BloomTransformerLayerInfer(TransformerLayerInferTpl):
    attention_uses_alibi = True

    """ """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.eps_ = network_config["layer_norm_epsilon"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.tp_world_size_
        self.tp_k_head_num_ = self.tp_q_head_num_
        self.tp_v_head_num_ = self.tp_q_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["n_embed"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["n_embed"]
        return

    def _context_attention_kernel(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        infer_state: InferStateInfo,
        layer_weight: BloomTransformerLayerWeight,
        out=None,
    ) -> torch.Tensor:
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        o_tensor = self.prefill_attention.forward(
            infer_state.attention_contexts[self.prefill_attention.group_id], DenseInputs(_q)
        )
        o_tensor = o_tensor.view(q.shape)
        return o_tensor

    def _token_attention_kernel(
        self, q: torch.Tensor, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight, out=None
    ) -> torch.Tensor:
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        o_tensor = self.decode_attention.forward(
            infer_state.attention_contexts[self.decode_attention.group_id], DenseInputs(_q)
        )
        return o_tensor.view(q.shape)

    def _att_norm(
        self, input: torch.Tensor, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.att_norm_weight_(
            input=input.view(-1, self.embed_dim_), eps=self.eps_, alloc_func=self.alloc_tensor
        )

    def _ffn_norm(
        self, input: torch.Tensor, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.ffn_norm_weight_(
            input=input.view(-1, self.embed_dim_), eps=self.eps_, alloc_func=self.alloc_tensor
        )

    def _get_qkv(
        self, input, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = self._tpsp_allgather(input, infer_state)
        q = layer_weight.q_proj.mm(input.view(-1, self.embed_dim_))
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        assert infer_state.need_dp_prefill_balance is False, "bloom does not support dp prefill balance"
        return q, cache_kv

    def _get_o(self, input, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight) -> torch.Tensor:
        assert infer_state.need_dp_prefill_balance is False, "bloom does not support dp prefill balance"
        o_tensor = layer_weight.o_proj.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_))
        o_tensor = self._tpsp_reduce(o_tensor, infer_state)
        return o_tensor

    def _ffn(self, input, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        ffn1_out = layer_weight.gate_up_proj.mm(input)
        input = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate="tanh")
        ffn1_out = None
        ffn2_out = layer_weight.down_proj.mm(gelu_out)
        gelu_out = None
        ffn2_out = self._tpsp_reduce(ffn2_out, infer_state)
        return ffn2_out

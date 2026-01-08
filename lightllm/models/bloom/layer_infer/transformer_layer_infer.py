import torch
from typing import Tuple
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.common.basemodel import InferStateInfo
from lightllm.common.basemodel.attention.base_att import AttControl


class BloomTransformerLayerInfer(TransformerLayerInferTpl):
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
        _k, _v = infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_)
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        o_tensor = infer_state.prefill_att_state.prefill_att(
            q=_q,
            k=_k,
            v=_v,
            att_control=AttControl(use_alibi=True, tp_alibi=layer_weight.tp_alibi),
            alloc_func=self.alloc_tensor,
        )
        o_tensor = o_tensor.view(q.shape)
        return o_tensor

    def _token_attention_kernel(
        self, q: torch.Tensor, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight, out=None
    ) -> torch.Tensor:
        _k, _v = infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_)
        _q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        o_tensor = infer_state.decode_att_state.decode_att(
            q=_q,
            k=_k,
            v=_v,
            att_control=AttControl(use_alibi=True, tp_alibi=layer_weight.tp_alibi),
            alloc_func=self.alloc_tensor,
        )
        return o_tensor.view(q.shape)

    def _att_norm(
        self, input: torch.Tensor, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.att_norm_weight_.layernorm_forward(
            input=input.view(-1, self.embed_dim_), eps=self.eps_, alloc_func=self.alloc_tensor
        )

    def _ffn_norm(
        self, input: torch.Tensor, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.ffn_norm_weight_.layernorm_forward(
            input=input.view(-1, self.embed_dim_), eps=self.eps_, alloc_func=self.alloc_tensor
        )

    def _get_qkv(
        self, input, infer_state: InferStateInfo, layer_weight: BloomTransformerLayerWeight
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = layer_weight.q_proj.mm(input.view(-1, self.embed_dim_))
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        return q, cache_kv

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

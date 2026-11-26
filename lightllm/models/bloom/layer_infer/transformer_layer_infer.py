import torch
from typing import Tuple
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs, gqa_kv_quant_fields
from lightllm.common.attention_state.full_att import FullAttentionState
from lightllm.common.basemodel.attention_backend_manager import LayerAttentionConfig, attention_is_causal
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.common.basemodel import InferStateInfo


class BloomTransformerLayerInfer(TransformerLayerInferTpl):
    """ """

    prefill_backend = None
    decode_backend = None

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

    def _gqa_cache_state(self, manager, model_role):
        return manager.get_state(
            model_role=model_role,
            state_type_filter="gqa",
            logical_layer_id=self.layer_num_,
            state_class=FullAttentionState,
        )

    def _gqa_config(self, model, *, has_alibi=False):
        state = self._gqa_cache_state(model.att_state_manager, model.att_state_role)
        keys, _ = state.get_layer_view(logical_layer_id=self.layer_num_)
        scales = state.get_layer_kv_scales(self.layer_num_)
        return GQALayerConfig(
            num_heads=self.tp_q_head_num_,
            num_kv_heads=keys.shape[1],
            head_dim=self.head_dim_,
            dtype=model.data_type,
            causal=attention_is_causal(model),
            has_alibi=has_alibi,
            **gqa_kv_quant_fields(model.args.llm_kv_type, scales, self.head_dim_),
        )

    def attention_config(self, model) -> LayerAttentionConfig:
        config = self._gqa_config(model, has_alibi=True)
        return LayerAttentionConfig(
            prefill=config, decode=config, prefill_backend_name="triton", decode_backend_name="triton"
        )

    def _gqa_alibi_slopes(self, layer_weight: BloomTransformerLayerWeight):
        return layer_weight.tp_alibi

    def _gqa_attention(self, backend, q, infer_state, layer_weight):
        state = self._gqa_cache_state(infer_state.att_state_manager, infer_state.att_state_role)
        keys, values = state.get_layer_view(logical_layer_id=self.layer_num_)
        config = backend.layer_config
        query = q.view(-1, config.num_heads, config.head_dim)
        inputs = GQALayerInputs(
            q=query,
            k_cache=keys,
            v_cache=values,
            kv_scales=state.get_layer_kv_scales(self.layer_num_),
            alibi_slopes=self._gqa_alibi_slopes(layer_weight),
        )
        output = backend.forward(
            infer_state.att_states[backend.state_key],
            inputs,
            out=self.alloc_tensor(query.shape, dtype=query.dtype, device=query.device),
            alloc_func=self.alloc_tensor,
        )
        return output.view(q.shape)

    def _context_attention_kernel(self, q, kv, infer_state, layer_weight, out=None) -> torch.Tensor:
        return self._gqa_attention(self.prefill_backend, q, infer_state, layer_weight)

    def _token_attention_kernel(self, q, infer_state, layer_weight, out=None) -> torch.Tensor:
        return self._gqa_attention(self.decode_backend, q, infer_state, layer_weight)

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

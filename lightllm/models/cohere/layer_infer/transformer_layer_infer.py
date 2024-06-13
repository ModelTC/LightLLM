import torch
from functools import partial

from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_cohere_template import (
    TransformerLayerCohereInferTpl,
)
from lightllm.models.cohere.infer_struct import CohereInferStateInfo
from lightllm.models.cohere.layer_weights.transformer_layer_weight import CohereTransformerLayerWeight
from lightllm.models.cohere.triton_kernels.layernorm import layernorm_forward, torch_layernorm
from lightllm.models.cohere.triton_kernels.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd


class CohereTransformerLayerInfer(TransformerLayerCohereInferTpl):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_v_head_num_ = network_config["num_key_value_heads"] // self.world_size_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        self.eps_ = self.network_config_["layer_norm_eps"]
        self.use_qk_norm_ = network_config.get("use_qk_norm", False)
        self._bind_func()

    def _bind_func(self):
        self._bind_rotary_emb_fwd()
        self._bind_norm()
        self._bind_attn()

    def _rotary_emb_fwd(self, q, kv, position_cos, position_sin):
        return rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            kv,
            position_cos,
            position_sin,
        )

    def _bind_rotary_emb_fwd(self):
        self._rotary_emb_fwd = partial(CohereTransformerLayerInfer._rotary_emb_fwd, self)

    def _att_norm(self, input, infer_state, layer_weight):
        return layernorm_forward(input.unsqueeze(1), layer_weight.att_norm_weight_.unsqueeze(0), self.eps_).squeeze(1)

    def _q_norm(self, input, infer_state, layer_weight):
        return layernorm_forward(input, layer_weight.q_norm_weight_, self.eps_)

    def _k_norm(self, input, infer_state, layer_weight):
        return layernorm_forward(input, layer_weight.k_norm_weight_, self.eps_)

    def _bind_norm(self):
        self._att_norm = partial(CohereTransformerLayerInfer._att_norm, self)
        self._q_norm = partial(CohereTransformerLayerInfer._q_norm, self)
        self._k_norm = partial(CohereTransformerLayerInfer._k_norm, self)

    def _bind_attn(self):
        # no need to re-impl
        LlamaTransformerLayerInfer._bind_attention(self)

    def _get_o(
        self, input, infer_state: CohereInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        return o_tensor

    def _ffn(
        self, input, infer_state: CohereInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        up_gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_up_proj)
        ffn1_out = silu_and_mul_fwd(up_gate_out)
        input = None
        up_gate_out = None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None
        return ffn2_out

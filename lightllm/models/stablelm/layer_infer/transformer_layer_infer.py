import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from functools import partial

from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward
from lightllm.models.stablelm.layer_weights.transformer_layer_weight import StablelmTransformerLayerWeight
from lightllm.models.internlm.layer_infer.transformer_layer_infer import InternlmTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class StablelmTransformerLayerInfer(InternlmTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.partial_rotary_factor = self.network_config_.get("partial_rotary_factor", 1)
        return
    
    def _bind_norm(self):
        self._att_norm = partial(StablelmTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(StablelmTransformerLayerInfer._ffn_norm, self)
        return

    def _get_qkv(
        self, input, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: StablelmTransformerLayerWeight
    ) -> torch.Tensor:
        q = torch.addmm(
            layer_weight.q_bias_, input.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0
        )
        torch.addmm(
            layer_weight.kv_bias_,
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            beta=1.0,
            alpha=1.0,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            self.partial_rotary_factor,
        )
        return q, cache_kv

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: StablelmTransformerLayerWeight
    ) -> torch.Tensor:
        o_tensor = torch.mm(
            input.view(-1, self.tp_o_head_num_ * self.head_dim_),
            layer_weight.o_weight_,
        )
        return o_tensor

    def _att_norm(self, input, infer_state: LlamaInferStateInfo, layer_weight: StablelmTransformerLayerWeight) -> torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.att_norm_weight_,
            bias=layer_weight.att_norm_bias_,
            eps=self.eps_,
        )

    def _ffn_norm(self, input, infer_state: LlamaInferStateInfo, layer_weight: StablelmTransformerLayerWeight) -> torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.ffn_norm_weight_,
            bias=layer_weight.ffn_norm_bias_,
            eps=self.eps_,
        )
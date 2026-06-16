import torch
from typing import Tuple

from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import (
    Qwen3NextTransformerLayerInfer,
)
from lightllm.models.qwen3_5.layer_weights.transformer_layer_weight import (
    Qwen35TransformerLayerWeight,
)
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Qwen35TransformerLayerInfer(Qwen3NextTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        # Initialize mrope section from config
        rope_scaling = network_config.get("rope_scaling", {})
        mrope_section = rope_scaling.get("mrope_section", [11, 11, 10])
        self.mrope_section = torch.tensor(mrope_section, dtype=torch.int32, device="cuda")

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen35TransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)

        qkvo_gate_proj = getattr(layer_weight, "qkvo_gate_proj", None)
        if qkvo_gate_proj is None:
            qkv_out = layer_weight.qkv_proj.mm(input)
            o_gate = layer_weight._o_gate_proj.mm(input)
        else:
            qkv_gate_out = qkvo_gate_proj.mm(input)
            qkv_out, o_gate = qkv_gate_out.split(
                [
                    self.tp_q_head_num_ * self.head_dim_ + (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_,
                    self.tp_q_head_num_ * self.head_dim_,
                ],
                dim=-1,
            )
        q, cache_kv = qkv_out.split(
            [self.tp_q_head_num_ * self.head_dim_, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_], dim=-1
        )

        infer_state.gate_value = o_gate
        layer_weight.qk_norm_weight_(
            q,
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )
        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        mrope_triton_fused(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            self.mrope_section,
            is_interleaved=True,  # Qwen3 uses interleaved mrope
            partial_rotary_factor=self.partial_rotary_factor,
        )
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv

import torch
import torch.distributed as dist
from typing import Tuple

from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import (
    Qwen3NextFullAttentionTransformerLayerInfer,
    Qwen3NextGatedDeltaNetTransformerLayerInfer,
)
from lightllm.models.qwen3next.layer_weights.transformer_layer_weight import (
    Qwen3NextFullAttentionTransformerLayerWeight,
    Qwen3NextGatedDeltaNetTransformerLayerWeight,
)
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen35FullAttentionTransformerLayerInfer(Qwen3NextFullAttentionTransformerLayerInfer):
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
        layer_weight: Qwen3NextFullAttentionTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)

        qkv_out = layer_weight.qkv_proj.mm(input)
        q, cache_kv = qkv_out.split(
            [self.tp_q_head_num_ * self.head_dim_, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_], dim=-1
        )
        o_gate = layer_weight._o_gate_proj.mm(input)

        # In-place sigmoid for gate
        infer_state.gate_value = o_gate.sigmoid_()
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
        return q, cache_kv


class Qwen35GatedDeltaNetTransformerLayerInfer(Qwen3NextGatedDeltaNetTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        rope_scaling = network_config.get("rope_scaling", {})
        mrope_section = rope_scaling.get("mrope_section", [11, 11, 10])
        self.mrope_section = torch.tensor(mrope_section, dtype=torch.int32, device="cuda")

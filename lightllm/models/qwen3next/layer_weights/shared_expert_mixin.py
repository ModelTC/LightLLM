"""Mixin for shared expert weight initialization in Qwen3Next models."""

from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
)


class SharedExpertWeightMixin:
    """
    Mixin class providing shared expert weight initialization.

    Expects the following attributes to be set by the using class:
    - layer_num_: int
    - data_type_: torch.dtype
    - quant_cfg: Optional quantization config
    """

    def _init_gate_shared_expert_weight(self):
        """Initialize shared expert weights (gate_up, down, and gating)."""
        prefix = f"model.layers.{self.layer_num_}.mlp.shared_expert"
        self.shared_expert_gate_up_proj = ROWMMWeight(
            weight_names=[f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"],
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate_up_proj",
        )
        self.shared_expert_down_proj = COLMMWeight(
            weight_names=f"{prefix}.down_proj.weight",
            data_type=self.data_type_,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_down_proj",
        )
        self.shared_expert_gate = ROWMMWeight(
            weight_names=f"model.layers.{self.layer_num_}.mlp.shared_expert_gate.weight",
            data_type=self.data_type_,
            bias_names=None,
            quant_cfg=self.quant_cfg,
            layer_num=self.layer_num_,
            name="shared_expert_gate",
            tp_rank=0,
            tp_world_size=1,
        )

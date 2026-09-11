from __future__ import annotations

import torch

from lightllm.models.pi0.model import Pi0ActionExpertModel, Pi0VLMModel
from lightllm.models.pi05.layer_infer import (
    Pi05ActionPostLayerInfer,
    Pi05ActionPreLayerInfer,
    Pi05ActionTransformerLayerInfer,
)
from lightllm.models.pi05.layer_weights import (
    Pi05ActionPreAndPostLayerWeight,
    Pi05ActionTransformerLayerWeight,
)
from lightllm.models.registry import GenerationModelRegistry, ModelRegistry


@ModelRegistry("pi05", is_multimodal=True)
class Pi05VLMModel(Pi0VLMModel):
    """Pi0.5 PaliGemma prefix; the VLM architecture is shared with Pi0."""


@GenerationModelRegistry("pi05", output_modality="action")
class Pi05ActionExpertModel(Pi0ActionExpertModel):
    """Pi0.5 controller with discrete prefix state and adaptive RMSNorm."""

    pre_layer_infer_class = Pi05ActionPreLayerInfer
    post_layer_infer_class = Pi05ActionPostLayerInfer
    transformer_layer_infer_class = Pi05ActionTransformerLayerInfer
    pre_and_post_weight_class = Pi05ActionPreAndPostLayerWeight
    transformer_weight_class = Pi05ActionTransformerLayerWeight
    first_suffix_token_isolated = False
    action_weight_prefixes = (
        "action_in_proj.",
        "action_out_proj.",
        "time_mlp_",
        "paligemma_with_expert.gemma_expert.model.norm.",
    )

    def _prepare_state(self, state: torch.Tensor | None) -> None:
        # Pi0.5 already placed discretized state tokens in the immutable prefix.
        return None

    def _suffix_length(self, action_horizon: int) -> int:
        return action_horizon

    def _denoise_step(
        self,
        infer_state,
        noisy_actions: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, condition = self.pre_infer.context_forward(
            noisy_actions,
            time_embedding,
            self.pre_post_weight,
        )
        for layer, weight in zip(
            self.layers_infer, self.trans_layers_weight, strict=True
        ):
            hidden_states = layer.context_forward(
                hidden_states,
                condition,
                infer_state,
                weight,
            )
        return self.post_infer.token_forward(
            hidden_states,
            condition,
            infer_state.action_horizon,
            self.pre_post_weight,
        )

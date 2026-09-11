import torch

from lightllm.common.basemodel import PostLayerInferTpl
from lightllm.models.pi05.layer_infer.transformer_layer_infer import (
    adaptive_rmsnorm,
)
from lightllm.models.pi05.layer_weights import Pi05ActionPreAndPostLayerWeight


class Pi05ActionPostLayerInfer(PostLayerInferTpl):
    def token_forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        action_horizon: int,
        weight: Pi05ActionPreAndPostLayerWeight,
    ) -> torch.Tensor:
        hidden_states, _ = adaptive_rmsnorm(
            hidden_states,
            condition,
            weight.final_norm_dense,
            eps=1e-6,
            alloc_tensor=self.alloc_tensor,
        )
        batch_size = condition.shape[0]
        actions = weight.action_out_proj.mm(
            hidden_states.reshape(-1, hidden_states.shape[-1])
        ).view(batch_size, action_horizon, -1)
        return actions.float()

import torch
import torch.nn.functional as F

from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.models.pi05.layer_weights import Pi05ActionPreAndPostLayerWeight


class Pi05ActionPreLayerInfer(PreLayerInferTpl):
    """Build Pi0.5 action tokens and the timestep condition once per step."""

    def context_forward(
        self,
        noisy_actions: torch.Tensor,
        time_embedding: torch.Tensor,
        weight: Pi05ActionPreAndPostLayerWeight,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_actions = noisy_actions.to(weight.data_type_)
        batch_size, action_horizon = noisy_actions.shape[:2]
        action_tokens = weight.action_in_proj.mm(
            noisy_actions.reshape(-1, noisy_actions.shape[-1])
        )
        condition = F.silu(weight.time_mlp_in.mm(time_embedding))
        condition = F.silu(weight.time_mlp_out.mm(condition))
        return action_tokens, condition

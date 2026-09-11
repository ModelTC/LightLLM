from __future__ import annotations

import torch
import torch.nn.functional as F

from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.models.pi0.layer_weights.pre_and_post_layer_weight import (
    Pi0ActionPreAndPostLayerWeight,
)


class Pi0ActionPreLayerInfer(PreLayerInferTpl):
    def context_forward(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        time_embedding: torch.Tensor,
        weight: Pi0ActionPreAndPostLayerWeight,
    ) -> torch.Tensor:
        # The model forward has already adapted request dimensions to the fixed
        # checkpoint projection widths. Keep this per-denoise-step path reshape-only.
        noisy_actions = noisy_actions.to(weight.data_type_)
        batch_size, action_horizon = noisy_actions.shape[:2]
        action_tokens = weight.action_in_proj.mm(
            noisy_actions.reshape(-1, noisy_actions.shape[-1])
        ).view(batch_size, action_horizon, -1)
        # Pi0 keeps the continuous observation state in the action suffix:
        #   state [B, state_dim] -> one state token [B, 1, hidden]
        #   noisy actions + timestep -> action tokens [B, horizon, hidden]
        # They are concatenated on the sequence dimension as [state, actions...].
        state_token = weight.state_proj.mm(state)[:, None, :]
        time_tokens = time_embedding[:, None, :].expand_as(action_tokens)
        action_tokens = torch.cat((action_tokens, time_tokens), dim=-1)
        action_tokens = F.silu(
            weight.time_mlp_in.mm(action_tokens.reshape(-1, action_tokens.shape[-1]))
        )
        action_tokens = weight.time_mlp_out.mm(action_tokens).view(
            batch_size, action_horizon, -1
        )

        return torch.cat((state_token, action_tokens), dim=1).view(
            batch_size * (action_horizon + 1), -1
        )

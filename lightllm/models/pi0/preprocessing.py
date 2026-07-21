from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

from .config import Pi0VLAConfig


class NormalizationMode(str, Enum):
    IDENTITY = "identity"
    MEAN_STD = "mean_std"
    QUANTILES = "quantiles"


@dataclass
class NormalizationSpec:
    mode: NormalizationMode = NormalizationMode.IDENTITY
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    q01: Optional[torch.Tensor] = None
    q99: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, value: Optional[dict[str, Any]]) -> "NormalizationSpec":
        if not value:
            return cls()
        mode = str(value.get("mode", value.get("type", "identity"))).lower()
        aliases = {"meanstd": "mean_std", "zscore": "mean_std", "quantile": "quantiles"}
        result = cls(mode=NormalizationMode(aliases.get(mode, mode)))
        for name in ("mean", "std", "q01", "q99"):
            if value.get(name) is not None:
                setattr(result, name, torch.as_tensor(value[name], dtype=torch.float32))
        if result.mode is NormalizationMode.MEAN_STD and (result.mean is None or result.std is None):
            raise ValueError("mean_std normalization requires mean and std")
        if result.mode is NormalizationMode.QUANTILES and (result.q01 is None or result.q99 is None):
            raise ValueError("quantile normalization requires q01 and q99")
        return result

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        dim = value.shape[-1]
        if self.mode is NormalizationMode.IDENTITY:
            return value
        if self.mode is NormalizationMode.MEAN_STD:
            mean = self.mean[..., :dim].to(value.device)
            std = self.std[..., :dim].to(value.device)
            return (value - mean) / (std + 1e-6)
        q01 = self.q01[..., :dim].to(value.device)
        q99 = self.q99[..., :dim].to(value.device)
        return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def unnormalize(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        dim = value.shape[-1]
        if self.mode is NormalizationMode.IDENTITY:
            return value
        if self.mode is NormalizationMode.MEAN_STD:
            stats_dim = self.mean.shape[-1]
            handled_dim = min(dim, stats_dim)
            # OpenPI pads missing means with 0 and stds with 1, then applies
            # the same epsilon to every dimension.
            output = value * (1.0 + 1e-6)
            mean = self.mean[..., :handled_dim].to(value.device)
            std = self.std[..., :handled_dim].to(value.device)
            output[..., :handled_dim] = value[..., :handled_dim] * (std + 1e-6) + mean
            return output
        stats_dim = self.q01.shape[-1]
        handled_dim = min(dim, stats_dim)
        output = value.clone()
        q01 = self.q01[..., :handled_dim].to(value.device)
        q99 = self.q99[..., :handled_dim].to(value.device)
        output[..., :handled_dim] = (value[..., :handled_dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        return output


class Pi0PrePostProcessor:
    """OpenPI-compatible state normalization and action postprocessing."""

    def __init__(self, config: Pi0VLAConfig):
        norm = config.norm_config or {}
        state_config = norm.get("state", norm.get("observation.state"))
        action_config = norm.get("action", norm.get("actions"))
        self.state_normalizer = NormalizationSpec.from_dict(state_config)
        self.action_normalizer = NormalizationSpec.from_dict(action_config)
        self.robot_adapter = config.robot_adapter or {}
        self.action_postprocess_config = config.action_postprocess_config or {}

    def normalize_state(self, state) -> torch.Tensor:
        return self.state_normalizer.normalize(torch.as_tensor(state, dtype=torch.float32))

    def postprocess_actions(self, actions: torch.Tensor, raw_state=None) -> torch.Tensor:
        result = self.action_normalizer.unnormalize(actions)
        relative_mask = self.robot_adapter.get("relative_action_mask")
        if relative_mask is not None and raw_state is not None:
            mask = torch.as_tensor(relative_mask, dtype=torch.bool, device=result.device)
            state = torch.as_tensor(raw_state, dtype=result.dtype, device=result.device)
            dims = min(mask.numel(), result.shape[-1], state.shape[-1])
            if result.ndim == 2:
                state_row = state if state.ndim == 1 else state[0]
                offset = torch.where(
                    mask[:dims],
                    state_row[:dims],
                    torch.zeros_like(state_row[:dims]),
                )
            else:
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                if state.shape[0] != result.shape[0]:
                    raise ValueError("raw state batch does not match action batch")
                offset = torch.where(
                    mask[None, :dims],
                    state[:, :dims],
                    torch.zeros_like(state[:, :dims]),
                )[:, None, :]
            result[..., :dims] += offset
        if self.action_postprocess_config.get("scale") is not None:
            result = result * torch.as_tensor(
                self.action_postprocess_config["scale"],
                dtype=result.dtype,
                device=result.device,
            )
        if self.action_postprocess_config.get("offset") is not None:
            result = result + torch.as_tensor(
                self.action_postprocess_config["offset"],
                dtype=result.dtype,
                device=result.device,
            )
        if self.action_postprocess_config.get("clip") is not None:
            lower, upper = self.action_postprocess_config["clip"]
            result = result.clamp(lower, upper)
        return result

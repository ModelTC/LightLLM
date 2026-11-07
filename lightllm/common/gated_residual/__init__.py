from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from .kernels import grouped_norm, inject_streams, injection_gates
from .projection import project_and_mix


@dataclass(frozen=True)
class GatedResidual:
    norm_weight: torch.Tensor
    down_weight: torch.Tensor
    up_weight: torch.Tensor
    inject_weight: Optional[torch.Tensor]
    branches: int
    eps: float = 1e-6

    def __post_init__(self):
        width = self.norm_weight.numel()
        if self.branches < 2 or width == 0 or width % self.branches:
            raise ValueError("Residual width must be a positive multiple of branches >= 2")
        if self.norm_weight.shape != (width,) or self.down_weight.ndim != 2:
            raise ValueError("Expected a norm vector and a matrix for the down projection")
        rank = self.down_weight.shape[0]
        if rank == 0 or self.down_weight.shape[1] != width or self.up_weight.shape != (width, rank):
            raise ValueError("Residual projection dimensions do not match")
        if self.inject_weight is not None and self.inject_weight.shape != (self.branches, width):
            raise ValueError("Injection projection must have shape (branches, residual width)")
        weights = [self.norm_weight, self.down_weight, self.up_weight]
        if self.inject_weight is not None:
            weights.append(self.inject_weight)
        if self.eps <= 0:
            raise ValueError("Normalization epsilon must be positive")
        for weight in weights:
            if not weight.is_cuda or not weight.is_contiguous():
                raise ValueError("Gated residual weights must be contiguous CUDA tensors")
            if weight.device != self.norm_weight.device or weight.dtype != self.norm_weight.dtype:
                raise ValueError("Gated residual weights must share a device and dtype")
        if self.norm_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("Gated residuals require floating point weights")

    @classmethod
    def from_checkpoint(
        cls,
        weights: Mapping[str, torch.Tensor],
        prefix: str,
        branches: int,
        eps: float = 1e-6,
        use_combine: bool = True,
    ) -> "GatedResidual":
        return cls(
            weights[f"{prefix}.hc_norm.weight"],
            weights[f"{prefix}.input_mix_weight_down.weight"],
            weights[f"{prefix}.input_mix_weight_up.weight"],
            weights[f"{prefix}.block_inject_weight.weight"] if use_combine else None,
            branches,
            eps,
        )

    def _check_input(self, x: torch.Tensor, width: int):
        if x.ndim < 2 or x.shape[-1] != width or not x.is_contiguous():
            raise ValueError(f"Expected contiguous input with last dimension {width}")
        if x.device != self.norm_weight.device or x.dtype != self.norm_weight.dtype:
            raise ValueError("Input must share the weights' device and dtype")

    @torch.no_grad()
    def read(self, residual: torch.Tensor):
        self._check_input(residual, self.norm_weight.numel())
        normalized = grouped_norm(residual, self.norm_weight, self.branches, self.eps)
        mixed = project_and_mix(normalized, self.down_weight, self.up_weight, self.branches)
        if self.inject_weight is None:
            return mixed
        gates = injection_gates(normalized, self.inject_weight, self.branches)
        return mixed, gates

    @torch.no_grad()
    def write(self, residual: torch.Tensor, update: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        width = self.norm_weight.numel()
        self._check_input(residual, width)
        self._check_input(update, width // self.branches)
        self._check_input(gates, self.branches)
        if update.shape[:-1] != residual.shape[:-1] or gates.shape[:-1] != residual.shape[:-1]:
            raise ValueError("Residual, update, and gates must have matching token dimensions")
        return inject_streams(residual, update, gates)

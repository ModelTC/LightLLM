from dataclasses import dataclass
from typing import Any, Protocol

import torch


class RmsNorm(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class GatedRmsNorm(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        eps: float,
        z: torch.Tensor,
        out: torch.Tensor | None = None,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        run_config: dict[str, Any] | None = None,
    ) -> torch.Tensor: ...


class LayerNorm(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
    ) -> torch.Tensor: ...


class QkRmsNorm(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor: ...


class FusedQkRmsNorm(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        eps: float = 1e-6,
        fp32_multiply: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class NormOps:
    rms_norm: RmsNorm
    gated_rms_norm: GatedRmsNorm
    layer_norm: LayerNorm
    qk_rms_norm: QkRmsNorm
    fused_qk_rms_norm: FusedQkRmsNorm

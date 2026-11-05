from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal
from lightllm.common.execution.contracts.state import FormatSpec
from lightllm.common.execution.contracts.device import ExecutionLimits
from lightllm.common.execution.compute.linear.spec import LinearSpec
from lightllm.common.execution.compute.moe.spec import MoeSpec

Phase = Literal["prefill", "decode", "verify"]


@dataclass(frozen=True)
class AttentionOffer:
    backend: str
    phase: Phase
    format: FormatSpec
    workspace_bytes: int = 0
    # Only implementations whose layers execute serially may share this scratch.
    # Concurrent microbatches occupy separate slices within the reservation.
    workspace_group: str | None = None
    execution_limits: ExecutionLimits | None = None
    transfer_requires_compute_sync: bool = False

    def __post_init__(self):
        if not self.backend or self.phase not in ("prefill", "decode", "verify") or self.workspace_bytes < 0:
            raise ValueError("invalid attention offer")


@dataclass(frozen=True)
class LinearOffer:
    backend: str
    spec: LinearSpec
    format: FormatSpec


@dataclass(frozen=True)
class MoeOffer:
    backend: str
    spec: MoeSpec
    format: FormatSpec
    fused_shared_experts: int = 0


@dataclass(frozen=True)
class ProbeResult:
    offers: tuple[AttentionOffer | LinearOffer | MoeOffer, ...] = ()
    reason: str = ""

    def __post_init__(self):
        if bool(self.offers) == bool(self.reason):
            raise ValueError("probe must return either offers or a rejection reason")

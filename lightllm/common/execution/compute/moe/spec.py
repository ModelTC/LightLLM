from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal


@dataclass(frozen=True)
class MoeSpec:
    hidden_size: int
    intermediate_size: int
    experts: int
    top_k: int
    dtype: Any
    renormalize: bool = False
    scoring: str = "softmax"
    groups: int = 1
    selected_groups: int = 1
    group_score_topk: int = 1
    routed_scale: float = 1.0
    shared_experts: int = 0
    gated_shared: bool = False
    alpha: float = 1.0
    limit: float | None = None
    clamp_up_add_one: bool = True
    # Preserve input-dtype rounding in routing, projections, activations and routed/shared subtotals.
    round_intermediates: bool = False

    def __post_init__(self):
        if min(self.hidden_size, self.intermediate_size, self.experts, self.top_k, self.groups) <= 0:
            raise ValueError("MoE dimensions and routing counts must be positive")
        if self.experts % self.groups or not 0 < self.selected_groups <= self.groups:
            raise ValueError("MoE groups must partition the experts")
        if self.top_k > self.selected_groups * (self.experts // self.groups):
            raise ValueError("MoE top-k exceeds selected group capacity")
        if not 0 < self.group_score_topk <= self.experts // self.groups:
            raise ValueError("MoE group score exceeds group capacity")
        if self.scoring not in ("softmax", "sigmoid", "topk_softmax"):
            raise ValueError("unsupported MoE scoring function")
        if self.scoring == "topk_softmax" and (self.groups != 1 or not self.renormalize):
            raise ValueError("top-k softmax requires ungrouped, normalized routing")
        if self.shared_experts < 0 or (self.gated_shared and not self.shared_experts):
            raise ValueError("gated MoE needs shared experts")
        if not isfinite(self.routed_scale) or not isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("MoE scales must be finite and activation alpha positive")
        if self.limit is not None and (not isfinite(self.limit) or self.limit <= 0):
            raise ValueError("MoE activation limit must be finite and positive")
        if self.limit is None and self.alpha != 1:
            raise ValueError("MoE activation alpha requires an explicit limit")

"""Immutable execution metadata and the attention-facing SSM updater contract."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass(frozen=True, slots=True)
class SSMRunConfig(Mapping[str, int]):
    """A detached kernel layout, also readable by existing Triton launch code."""

    BV: int
    num_warps: int
    num_stages: int = 3

    def __post_init__(self):
        for key in self:
            if type(self[key]) is not int:
                raise TypeError(f"SSM kernel configuration {key} must be an integer")

    @classmethod
    def from_mapping(cls, config: Mapping[str, int]) -> "SSMRunConfig":
        unknown = config.keys() - {"BV", "num_warps", "num_stages"}
        if unknown:
            raise ValueError(f"Unknown SSM kernel configuration fields: {unknown}")
        return cls(BV=config["BV"], num_warps=config["num_warps"], num_stages=config.get("num_stages", 3))

    def __getitem__(self, key: str) -> int:
        if key == "BV":
            return self.BV
        if key == "num_warps":
            return self.num_warps
        if key == "num_stages":
            return self.num_stages
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("BV", "num_warps", "num_stages"))

    def __len__(self) -> int:
        return 3


@dataclass(frozen=True, slots=True)
class RecurrentStateContext:
    """Per-forward/graph commit metadata, passed unchanged by the runner.

    Both verify and compact reconstruction read the same immutable layout.
    Publishing a context detaches it from the tuner's mutable dictionaries.
    """

    config: SSMRunConfig

    def __post_init__(self):
        if not isinstance(self.config, SSMRunConfig):
            raise TypeError("Use from_run_config() to freeze a mutable kernel configuration")

    @classmethod
    def from_run_config(cls, config: Mapping[str, int]) -> "RecurrentStateContext":
        return cls(SSMRunConfig.from_mapping(config))


class SSMUpdateBackend(Protocol):
    """Attention executes the updater directly; lifecycle stays with its owner.

    ``a`` and ``b`` are raw gate/beta projections. GDN uses scalar decay per
    value head; KDA's ``a`` and ``bias`` include the key dimension. A missing
    ``cu_seqlens`` denotes ordinary one-token-per-request decode.
    """

    def get_run_config(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> Mapping[str, int]:
        ...

    def forward(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
        reqs: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        run_config: Optional[Mapping[str, int]] = None,
    ) -> torch.Tensor:
        ...

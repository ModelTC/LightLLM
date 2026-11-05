from dataclasses import dataclass


@dataclass(frozen=True)
class RMSNormSpec:
    width: int
    eps: float


@dataclass(frozen=True)
class RotarySpec:
    head_dim: int
    interleaved: bool = False


@dataclass(frozen=True)
class SwiGLUSpec:
    hidden_size: int
    intermediate_size: int

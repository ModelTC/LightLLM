from dataclasses import dataclass

from lightllm.platform.ops.act import ActOps
from lightllm.platform.ops.embedding import EmbeddingOps
from lightllm.platform.ops.norm import NormOps
from lightllm.platform.ops.rotary import RotaryOps
from lightllm.platform.ops.sampling import SamplingOps


@dataclass(frozen=True)
class PlatformOps:
    norm: NormOps
    act: ActOps
    sampling: SamplingOps
    embedding: EmbeddingOps
    rotary: RotaryOps


__all__ = ["ActOps", "EmbeddingOps", "NormOps", "PlatformOps", "RotaryOps", "SamplingOps"]

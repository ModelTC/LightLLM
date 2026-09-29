from dataclasses import dataclass

from lightllm.platform.ops.act import ActOps
from lightllm.platform.ops.norm import NormOps
from lightllm.platform.ops.sampling import SamplingOps


@dataclass(frozen=True)
class PlatformOps:
    norm: NormOps
    act: ActOps
    sampling: SamplingOps


__all__ = ["ActOps", "NormOps", "PlatformOps", "SamplingOps"]

from dataclasses import dataclass
from typing import Any, Callable

import torch


TensorOp = Callable[..., torch.Tensor]
AnyOp = Callable[..., Any]


@dataclass(frozen=True)
class NormOps:
    rms_norm: TensorOp
    gated_rms_norm: TensorOp
    layer_norm: TensorOp
    qk_rms_norm: TensorOp
    fused_qk_rms_norm: AnyOp

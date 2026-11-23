from typing import Any, Optional

import torch

from .backend import AttentionImpl, TensorAllocator


class Attention:
    """One layer's attention: its prefill and decode impls and the layer group whose step args they read.

    A model creates it when it loads, so an unsupported layer config fails then rather than on the
    first forward. Layers with an equal layer config share impls. Each call passes the step args
    of the layer's group, infer_state.att_step_args[attention.group]. A phase's impl is None when the
    layer does not attend in that phase.
    """

    def __init__(self, group: str, prefill_impl: Optional[AttentionImpl], decode_impl: Optional[AttentionImpl]) -> None:
        self.group = group
        self.prefill_impl = prefill_impl
        self.decode_impl = decode_impl

    def prefill(
        self, step_args: Any, layer_inputs: Any, *, out: torch.Tensor, alloc_func: TensorAllocator = torch.empty
    ) -> torch.Tensor:
        if self.prefill_impl is None:
            raise RuntimeError(f"layers of attention group {self.group} do not attend in prefill")
        return self.prefill_impl.forward(step_args, layer_inputs, out=out, alloc_func=alloc_func)

    def decode(
        self, step_args: Any, layer_inputs: Any, *, out: torch.Tensor, alloc_func: TensorAllocator = torch.empty
    ) -> torch.Tensor:
        if self.decode_impl is None:
            raise RuntimeError(f"layers of attention group {self.group} do not attend in decode")
        return self.decode_impl.forward(step_args, layer_inputs, out=out, alloc_func=alloc_func)

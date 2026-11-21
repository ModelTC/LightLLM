import dataclasses
import weakref
from typing import Optional, Tuple

import torch

from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeStepArgs
from lightllm.common.attention.backends.mla.fa3 import (
    FA3MLADecodeBackend,
    FA3MLAPrefillBackend,
    FA3MLAPrefillStepArgs,
)
from ..base_att import BasePrefillAttState, BaseDecodeAttState, AttControl, build_attention_batch
from ..mla_layer import mla_decode_layer, mla_prefill_layer
from .fp import Fa3AttBackend


class MlaFa3AttBackend(Fa3AttBackend):
    """FA3 MLA; decode shares Fa3AttBackend's decode step builder, whose row grouping and page table fit MLA."""

    def __init__(self, model):
        super().__init__(model=model)
        self.mla_prefill_step_builder = FA3MLAPrefillBackend.get_step_builder_cls()()

    def create_att_prefill_state(self, infer_state) -> "MlaFa3PrefillAttState":
        return MlaFa3PrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "MlaFa3DecodeAttState":
        return MlaFa3DecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class MlaFa3PrefillAttState(BasePrefillAttState):
    step_args: Optional[FA3MLAPrefillStepArgs] = None
    causal: bool = None

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        batch = build_attention_batch(self.infer_state, self.backend)
        self.step_args = self.backend.mla_prefill_step_builder.build(batch)

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        layer_inputs, layer_config = mla_prefill_layer(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(FA3MLAPrefillBackend, layer_config)
        out = alloc_func((q.shape[0], q.shape[1], v.shape[-1]), dtype=q.dtype, device=q.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)


@dataclasses.dataclass
class MlaFa3DecodeAttState(BaseDecodeAttState):
    step_args: Optional[FA3GQADecodeStepArgs] = None
    causal: bool = None

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        # Replay steps also build: the step builder rewrites the tensors the graph captured.
        self.step_args = self.backend.decode_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        layer_inputs, layer_config = mla_decode_layer(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(FA3MLADecodeBackend, layer_config)
        q_nope = layer_inputs.q_nope
        out = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)

import dataclasses
import weakref
from typing import Optional

import torch

from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs
from lightllm.common.attention.backends.gqa.triton import (
    TritonGQADecodeBackend,
    TritonGQADecodeStepArgs,
    TritonGQAPrefillBackend,
    TritonGQAPrefillStepArgs,
)
from ..base_att import (
    BaseAttBackend,
    BasePrefillAttState,
    BaseDecodeAttState,
    AttControl,
    build_attention_batch,
)


def _gqa_layer_inputs_and_config(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, att_control: AttControl, causal: bool
):
    # Pass every attention property the model asks for, so the impl rejects what Triton cannot run.
    sliding_window = att_control.sliding_window if att_control.use_sliding_window else (-1, -1)
    layer_config = GQALayerConfig(
        num_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_dim=q.shape[2],
        dtype=q.dtype,
        sliding_window=sliding_window,
        causal=causal,
        has_sinks=att_control.use_att_sink,
        has_alibi=att_control.use_alibi,
    )
    alibi_slopes = att_control.tp_alibi if att_control.use_alibi else None
    return GQALayerInputs(q=q, k_cache=k, v_cache=v, alibi_slopes=alibi_slopes), layer_config


class TritonAttBackend(BaseAttBackend):
    def __init__(self, model):
        super().__init__(model)
        self.prefill_step_builder = TritonGQAPrefillBackend.get_step_builder_cls()()
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs read the tensors it keeps.
        if getattr(self, "decode_step_builder", None) is None:
            self.decode_step_builder = TritonGQADecodeBackend.get_step_builder_cls()()

    def create_att_prefill_state(self, infer_state) -> "TritonPrefillAttState":
        return TritonPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "TritonDecodeAttState":
        return TritonDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class TritonPrefillAttState(BasePrefillAttState):
    step_args: Optional[TritonGQAPrefillStepArgs] = None
    causal: bool = True

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        batch = build_attention_batch(self.infer_state, self.backend)
        self.step_args = self.backend.prefill_step_builder.build(batch)

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        layer_inputs, layer_config = _gqa_layer_inputs_and_config(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(TritonGQAPrefillBackend, layer_config)
        return impl.forward(
            self.step_args, layer_inputs, out=alloc_func(q.shape, dtype=q.dtype, device=q.device), alloc_func=alloc_func
        )


@dataclasses.dataclass
class TritonDecodeAttState(BaseDecodeAttState):
    step_args: Optional[TritonGQADecodeStepArgs] = None
    causal: bool = True

    def init_state(self):
        self.causal = self.backend.uses_causal_attention()
        # Replay steps also build: the step builder rewrites the tensors the graph captured.
        batch = build_attention_batch(self.infer_state, self.backend)
        self.step_args = self.backend.decode_step_builder.build(batch)

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        layer_inputs, layer_config = _gqa_layer_inputs_and_config(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(TritonGQADecodeBackend, layer_config)
        return impl.forward(
            self.step_args, layer_inputs, out=alloc_func(q.shape, dtype=q.dtype, device=q.device), alloc_func=alloc_func
        )

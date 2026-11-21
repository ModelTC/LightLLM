import dataclasses
import weakref
from typing import Optional

import torch

from lightllm.common.attention.backends.gqa import GQALayerConfig, GQALayerInputs
from lightllm.common.attention.backends.gqa.triton_quant import (
    TritonQuantGQADecodeBackend,
    TritonQuantGQADecodeStepArgs,
    TritonQuantGQAPrefillBackend,
    TritonQuantGQAPrefillStepArgs,
)
from lightllm.utils.envs_utils import enable_diverse_mode_gqa_decode_fast_kernel
from ..base_att import BaseAttBackend, BasePrefillAttState, BaseDecodeAttState, AttControl, build_attention_batch


class Int8kvTritonAttBackend(BaseAttBackend):
    """Triton attention over a group-quantized int8 KV cache; scales come per layer in AttControl.kv_scales."""

    kv_quant: str = "int8_group"

    def __init__(self, model):
        super().__init__(model)
        self.prefill_step_builder = TritonQuantGQAPrefillBackend.get_step_builder_cls()()
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs read the tensors it keeps.
        if getattr(self, "decode_step_builder", None) is None:
            self.decode_step_builder = TritonQuantGQADecodeBackend.get_step_builder_cls()(
                diverse=self._uses_diverse_decode()
            )

    def _uses_diverse_decode(self) -> bool:
        return enable_diverse_mode_gqa_decode_fast_kernel()

    def create_att_prefill_state(self, infer_state) -> "QuantTritonPrefillAttState":
        return QuantTritonPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "QuantTritonDecodeAttState":
        return QuantTritonDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


def _quant_layer_inputs_and_config(q, k, v, att_control: AttControl, kv_quant: str):
    assert att_control.use_alibi is False, "quantized Triton GQA does not support alibi"
    scales = att_control.kv_scales
    assert scales is not None, "a quantized KV cache needs AttControl.kv_scales"
    head_dim = q.shape[2]
    layer_config = GQALayerConfig(
        num_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_dim=head_dim,
        dtype=q.dtype,
        sliding_window=att_control.sliding_window if att_control.use_sliding_window else (-1, -1),
        has_sinks=att_control.use_att_sink,
        kv_quant=kv_quant,
        kv_quant_group_size=head_dim // scales.k_scale.shape[-1],
    )
    return GQALayerInputs(q=q, k_cache=k, v_cache=v, kv_scales=scales), layer_config


@dataclasses.dataclass
class QuantTritonPrefillAttState(BasePrefillAttState):
    step_args: Optional[TritonQuantGQAPrefillStepArgs] = None

    def init_state(self):
        self.step_args = self.backend.prefill_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        layer_inputs, layer_config = _quant_layer_inputs_and_config(q, k, v, att_control, self.backend.kv_quant)
        impl = self.backend.get_attention_impl(TritonQuantGQAPrefillBackend, layer_config)
        return impl.forward(
            self.step_args, layer_inputs, out=alloc_func(q.shape, dtype=q.dtype, device=q.device), alloc_func=alloc_func
        )


@dataclasses.dataclass
class QuantTritonDecodeAttState(BaseDecodeAttState):
    step_args: Optional[TritonQuantGQADecodeStepArgs] = None

    def init_state(self):
        # Replay steps also build: the step builder rewrites the group tensors the graph captured.
        self.step_args = self.backend.decode_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        layer_inputs, layer_config = _quant_layer_inputs_and_config(q, k, v, att_control, self.backend.kv_quant)
        impl = self.backend.get_attention_impl(TritonQuantGQADecodeBackend, layer_config)
        return impl.forward(
            self.step_args, layer_inputs, out=alloc_func(q.shape, dtype=q.dtype, device=q.device), alloc_func=alloc_func
        )


# Names kept for code that refers to the int8 states.
Int8kvTritonPrefillAttState = QuantTritonPrefillAttState
Int8kvTritonDecodeAttState = QuantTritonDecodeAttState

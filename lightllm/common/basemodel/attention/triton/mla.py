import dataclasses
import weakref
import torch
from ..base_att import (
    BaseAttBackend,
    BasePrefillAttState,
    BaseDecodeAttState,
    AttControl,
    build_attention_batch,
)
from typing import TYPE_CHECKING, Tuple
from ..mla_layer import mla_decode_layer, mla_prefill_layer
from lightllm.common.attention.backend import TensorAllocator
from lightllm.common.attention.backends.mla.triton import (
    TritonMLADecodeBackend,
    TritonMLADecodeStepArgs,
    TritonMLAPrefillBackend,
    TritonMLAPrefillStepArgs,
)

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


class MlaTritonAttBackend(BaseAttBackend):
    def __init__(self, model: "TpPartBaseModel") -> None:
        super().__init__(model=model)
        self.prefill_step_builder = TritonMLAPrefillBackend.get_step_builder_cls()()
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs read the schedule buffers it owns.
        if getattr(self, "decode_step_builder", None) is None:
            # One decode step builder serves all MLA layers of the model, which share these dimensions;
            # TritonMLADecodeImpl rejects step args built for a different configuration.
            self.decode_step_builder = TritonMLADecodeBackend.get_step_builder_cls()(
                num_heads=model.config["num_attention_heads"] // model.tp_world_size_,
                kv_lora_rank=model.kv_lora_rank,
                qk_rope_head_dim=model.qk_rope_head_dim,
                dtype=model.data_type,
            )

    def create_att_prefill_state(self, infer_state) -> "MlaTritonPrefillAttState":
        return MlaTritonPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "MlaTritonDecodeAttState":
        return MlaTritonDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class MlaTritonPrefillAttState(BasePrefillAttState):
    step_args: TritonMLAPrefillStepArgs | None = None
    causal: bool = True

    def init_state(self) -> None:
        self.causal = self.backend.uses_causal_attention()
        batch = build_attention_batch(self.infer_state, self.backend)
        self.step_args = self.backend.prefill_step_builder.build(batch)

    def prefill_att(
        self,
        q: torch.Tensor,
        k: Tuple[torch.Tensor, torch.Tensor],
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        layer_inputs, layer_config = mla_prefill_layer(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(TritonMLAPrefillBackend, layer_config)
        out = alloc_func((q.shape[0], q.shape[1], v.shape[-1]), dtype=q.dtype, device=q.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)


@dataclasses.dataclass
class MlaTritonDecodeAttState(BaseDecodeAttState):
    step_args: TritonMLADecodeStepArgs | None = None
    causal: bool = True

    def init_state(self) -> None:
        self.causal = self.backend.uses_causal_attention()
        # Replay steps also build: the step builder rewrites the captured step's split schedule in place.
        batch = build_attention_batch(self.infer_state, self.backend)
        self.step_args = self.backend.decode_step_builder.build(batch)

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        layer_inputs, layer_config = mla_decode_layer(q, k, v, att_control, self.causal)
        impl = self.backend.get_attention_impl(TritonMLADecodeBackend, layer_config)
        out = alloc_func(layer_inputs.q_nope.shape, dtype=layer_inputs.q_nope.dtype, device=layer_inputs.q_nope.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)

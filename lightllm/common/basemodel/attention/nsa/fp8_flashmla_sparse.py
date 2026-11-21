"""DeepSeek V3.2 sparse MLA over the packed FP8 latent cache, with FlashMLA prefill and decode.

The NSA indexer's per-step metadata lives on Deepseek3_2InferStateInfo; this adapter only runs attention.
"""

import dataclasses
import weakref
from typing import Optional, Tuple

import torch

from lightllm.common.attention.backends.sparse_mla.flashmla import (
    FlashMLASparseDecodeBackend,
    FlashMLASparseDecodeStepArgs,
    FlashMLASparsePrefillBackend,
    FlashMLASparsePrefillStepArgs,
)
from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState, build_attention_batch
from .sparse_mla_layer import sparse_mla_decode_layer, sparse_mla_prefill_layer


class NsaFlashMlaFp8SparseAttBackend(BaseAttBackend):
    kv_cache_format = "fp8_ds_mla"

    def __init__(self, model):
        super().__init__(model=model)
        self.prefill_step_builder = FlashMLASparsePrefillBackend.get_step_builder_cls()()
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs use the scheduler metadata it keeps.
        if getattr(self, "decode_step_builder", None) is None:
            self.decode_step_builder = FlashMLASparseDecodeBackend.get_step_builder_cls()()

    def create_att_prefill_state(self, infer_state) -> "NsaFlashMlaFp8SparsePrefillAttState":
        return NsaFlashMlaFp8SparsePrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state) -> "NsaFlashMlaFp8SparseDecodeAttState":
        return NsaFlashMlaFp8SparseDecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class NsaFlashMlaFp8SparsePrefillAttState(BasePrefillAttState):
    step_args: Optional[FlashMLASparsePrefillStepArgs] = None

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
        layer_inputs, layer_config = sparse_mla_prefill_layer(q, k, att_control, self.backend.kv_cache_format)
        impl = self.backend.get_attention_impl(FlashMLASparsePrefillBackend, layer_config)
        out = alloc_func((q.shape[0], q.shape[1], layer_config.kv_lora_rank), dtype=q.dtype, device=q.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)


@dataclasses.dataclass
class NsaFlashMlaFp8SparseDecodeAttState(BaseDecodeAttState):
    step_args: Optional[FlashMLASparseDecodeStepArgs] = None

    def init_state(self):
        self.step_args = self.backend.decode_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def decode_att(
        self,
        q: Tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        layer_inputs, layer_config = sparse_mla_decode_layer(q, k, att_control, self.backend.kv_cache_format)
        impl = self.backend.get_attention_impl(FlashMLASparseDecodeBackend, layer_config)
        q_nope = layer_inputs.q_nope
        out = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)

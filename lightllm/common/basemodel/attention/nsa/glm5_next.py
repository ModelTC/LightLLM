"""GLM-5 NoPE sparse MLA over a bf16 latent cache: FlashMLA prefill, FA3 decode.

The NSA indexer's per-step metadata lives on Glm5NextInferStateInfo; this adapter only runs attention.
"""

import dataclasses
import weakref
from typing import Optional

import torch

from lightllm.common.attention.backend import AttentionStepBuilder
from lightllm.common.attention.backends.sparse_mla.fa3 import FA3SparseMLADecodeBackend
from lightllm.common.attention.backends.sparse_mla.flashmla import FlashMLASparsePrefillBackend
from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState, build_attention_batch
from .sparse_mla_layer import sparse_mla_decode_layer, sparse_mla_prefill_layer


class Glm5NextSparseAttBackend(BaseAttBackend):
    kv_cache_format = "bf16"
    prefill_backend = FlashMLASparsePrefillBackend
    decode_backend = FA3SparseMLADecodeBackend

    def __init__(self, model):
        super().__init__(model=model)
        self.prefill_step_builder = self._create_prefill_step_builder(model)
        # BaseAttBackend reuses one instance per model but reruns __init__ on every construction.
        # Keep the decode step builder: captured CUDA graphs read the state it keeps.
        if getattr(self, "decode_step_builder", None) is None:
            self.decode_step_builder = self._create_decode_step_builder(model)

    def _create_prefill_step_builder(self, model) -> AttentionStepBuilder:
        return FlashMLASparsePrefillBackend.get_step_builder_cls()()

    def _create_decode_step_builder(self, model) -> AttentionStepBuilder:
        # The indexer selects whole pools of index_kpool keys plus the newest partial pool.
        return FA3SparseMLADecodeBackend.get_step_builder_cls()(
            topk=model.config["index_topk"], kpool=model.config["index_kpool"], causal=False
        )

    def create_att_prefill_state(self, infer_state):
        return Glm5NextSparsePrefillState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state):
        return Glm5NextSparseDecodeState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class Glm5NextSparsePrefillState(BasePrefillAttState):
    step_args: Optional[object] = None

    def init_state(self):
        self.step_args = self.backend.prefill_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def prefill_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        backend = self.backend
        layer_inputs, layer_config = sparse_mla_prefill_layer(q, k, att_control, backend.kv_cache_format)
        impl = backend.get_attention_impl(backend.prefill_backend, layer_config)
        out = alloc_func((q.shape[0], q.shape[1], layer_config.kv_lora_rank), dtype=q.dtype, device=q.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)


@dataclasses.dataclass
class Glm5NextSparseDecodeState(BaseDecodeAttState):
    step_args: Optional[object] = None

    def init_state(self):
        # Replay steps also build: the step builder rewrites the state the graph captured.
        self.step_args = self.backend.decode_step_builder.build(build_attention_batch(self.infer_state, self.backend))

    def decode_att(self, q, k, v, att_control=AttControl(), alloc_func=torch.empty):
        backend = self.backend
        layer_inputs, layer_config = sparse_mla_decode_layer(q, k, att_control, backend.kv_cache_format)
        impl = backend.get_attention_impl(backend.decode_backend, layer_config)
        q_nope = layer_inputs.q_nope
        out = alloc_func(q_nope.shape, dtype=q_nope.dtype, device=q_nope.device)
        return impl.forward(self.step_args, layer_inputs, out=out, alloc_func=alloc_func)

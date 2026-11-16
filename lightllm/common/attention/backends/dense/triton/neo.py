from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase
import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.dense.triton.backend import (
    TritonDensePrefillBackend,
    TritonDenseDecodeBackend,
    TritonPrefillContext,
)


@dataclasses.dataclass(slots=True)
class NeoTritonPrefillContext(TritonPrefillContext):
    pass


class TritonDenseNeoCommon:
    family = AttentionFamily.DENSE
    variant = "neo"

    provider = "triton"

    cache_encodings = frozenset({CacheEncoding.PLAIN})


class TritonDenseNeoPrefillBackend(TritonDenseNeoCommon, TritonDensePrefillBackend):
    phase = AttentionPhase.PREFILL

    def allocate_context(self, capacity, *, capture, workspace) -> NeoTritonPrefillContext:
        return NeoTritonPrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: NeoTritonPrefillContext):
        pass

    def forward(
        self, layer: LayerAttention, context: NeoTritonPrefillContext, inputs: DenseInputs, *, out=None
    ) -> torch.Tensor:
        q = inputs.query
        k = layer.resources.key
        v = layer.resources.value
        from lightllm.common.attention.backends.dense.triton.kernels.prefill.context_attention_fwd_neo import (
            context_attention_fwd_neo,
        )

        out = self.alloc_tensor(q.shape, q.dtype)
        context_attention_fwd_neo(
            q,
            k,
            v,
            out,
            context.batch.b_req_idx,
            context.batch.b_q_start_loc,
            context.batch.b_seq_len,
            context.batch.b_ready_cache_len,
            context.batch.max_q_seq_len,
            context.batch.req_to_token_indexs,
            context.batch.b_image_token_end,
        )
        return out


class TritonDenseNeoDecodeBackend(TritonDenseNeoCommon, TritonDenseDecodeBackend):
    phase = AttentionPhase.DECODE

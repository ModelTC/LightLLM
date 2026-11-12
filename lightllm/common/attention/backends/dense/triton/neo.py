import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import DenseInputs
from lightllm.common.attention.backends.dense.triton.backend import TritonAttBackend, TritonPrefillContext


@dataclasses.dataclass
class NeoTritonPrefillContext(TritonPrefillContext):
    pass


class NeoTritonAttBackend(TritonAttBackend):
    provider = "triton"
    cache_formats = frozenset({"None"})

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return NeoTritonPrefillContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        pass

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: DenseInputs, *, out=None) -> torch.Tensor:
        alloc_func = layer.alloc_tensor
        q = inputs.query
        k = layer.params.key
        v = layer.params.value
        from lightllm.common.attention.backends.dense.triton.kernels.prefill.context_attention_fwd_neo import (
            context_attention_fwd_neo,
        )

        out = alloc_func(q.shape, q.dtype)
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

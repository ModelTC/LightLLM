import dataclasses
import torch
from lightllm.common.attention.backends.dense.triton.backend import (
    TritonAttBackend,
    TritonPrefillPlan,
    TritonDecodePlan,
)
from lightllm.common.attention.api.backend import LayerBinding
from lightllm.common.attention.api.inputs import DenseInputs


@dataclasses.dataclass
class NeoTritonPrefillPlan(TritonPrefillPlan):
    pass


class NeoTritonAttBackend(TritonAttBackend):
    provider = "triton"
    cache_formats = frozenset({"None"})

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return NeoTritonPrefillPlan(capacity, capture, workspace)

    def _update_prefill(self, plan):
        pass

    def _forward_prefill(self, layer: LayerBinding, plan, inputs: DenseInputs, *, out=None) -> torch.Tensor:
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
            plan.batch.b_req_idx,
            plan.batch.b_q_start_loc,
            plan.batch.b_seq_len,
            plan.batch.b_ready_cache_len,
            plan.batch.max_q_seq_len,
            plan.batch.req_to_token_indexs,
            plan.batch.b_image_token_end,
        )
        return out

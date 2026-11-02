from lightllm.common.basemodel.attention.contracts import ResourceId
from lightllm.common.basemodel.attention.sparse import BoundSparse, SparseDecodeInput, SparsePrefillInput


def _binding(kv, params):
    return BoundSparse(
        ResourceId("main", 0), kv, params["softmax_scale"], params["kv_lora_rank"], params.get("qk_rope_head_dim", 0)
    )


def run_sparse_prefill(state, q, kv, params):
    inputs = SparsePrefillInput(
        q, params.get("topk_mem_indices"), params.get("topk_indices"), params.get("prefill_cache_kv")
    )
    return state.run(inputs, _binding(kv, params))


def run_sparse_decode(state, q, kv, params):
    return state.run(SparseDecodeInput(*q, params["topk_mem_indices"]), _binding(kv, params))

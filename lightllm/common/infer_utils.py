from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req_prefill


def init_req_to_token_indexes(
    req_to_token_indexs,
    b_req_idx,
    b_seq_len,
    b_ready_cache_len,
    b_start_loc,
    alloc_mem_index,
    max_q_seq_len,
    mem_manager=None,
):
    """
    Initialize req_to_token_indexs for a prefill batch.

    This function handles standard KV cache copying for NEW tokens. Models with
    additional buffer requirements (e.g., multi-buffer KV cache) should implement
    the PrefillHookProvider protocol to inject custom synchronization logic.

    The hook-based approach keeps the common module independent of model-specific
    details, eliminating cross-layer imports and hidden dependencies.

    Args:
        req_to_token_indexs: Request to token index mapping
        b_req_idx: Batch request indices
        b_seq_len: Batch sequence lengths
        b_ready_cache_len: Batch ready cache lengths (for prefix cache)
        b_start_loc: Batch start locations
        alloc_mem_index: Allocated memory indices
        max_q_seq_len: Maximum query sequence length
        mem_manager: Memory manager (can be MultiBufferMemoryManager)
    """
    # Standard KV cache copy for NEW tokens
    copy_kv_index_to_req_prefill(
        req_to_token_indexs=req_to_token_indexs,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        b_start_loc=b_start_loc,
        memindex=alloc_mem_index,
        max_q_seq_len=max_q_seq_len,
    )

    # Note: Model-specific buffer synchronization is handled by
    # the model's sync_prefill_buffers() hook, not here.
    # This keeps the common module independent of model details.

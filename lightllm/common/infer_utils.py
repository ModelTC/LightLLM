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
    old_indexer_ks_positions=None,
):
    # Step 1: Copy KV cache for NEW tokens (existing logic)
    copy_kv_index_to_req_prefill(
        req_to_token_indexs=req_to_token_indexs,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        b_start_loc=b_start_loc,
        memindex=alloc_mem_index,
        max_q_seq_len=max_q_seq_len,
    )

    # Step 2: Copy indexer_ks for CACHED tokens (DeepSeek v3.2 specific)
    # This ensures consistency between KV cache and indexer_ks buffers
    # when prefix cache is hit
    if (
        mem_manager is not None
        and hasattr(mem_manager, "indexer_ks_mem_manager")
        and old_indexer_ks_positions is not None
    ):

        _copy_cached_indexer_ks_to_new_positions(
            req_to_token_indexs=req_to_token_indexs,
            b_req_idx=b_req_idx,
            b_ready_cache_len=b_ready_cache_len,
            mem_manager=mem_manager,
            old_indexer_ks_positions=old_indexer_ks_positions,
        )


def _copy_cached_indexer_ks_to_new_positions(
    req_to_token_indexs,
    b_req_idx,
    b_ready_cache_len,
    mem_manager,
    old_indexer_ks_positions,
):
    """
    Copy cached tokens' indexer_ks from old positions to new positions.

    This function is called after copy_kv_index_to_req_prefill() has updated
    req_to_token_indexs to point to new contiguous positions. We need to copy
    indexer_ks data to match the KV cache layout.

    For each layer and each request with cached tokens:
    - Copy indexer_ks data from old positions to new positions
    - This ensures consistency when using extract_indexer_ks later
    """
    from lightllm.models.deepseek3_2.triton_kernel.copy_indexer_ks import copy_indexer_ks

    # Get number of layers from indexer_ks_mem_manager
    num_layers = len(mem_manager.indexer_ks_mem_manager.kv_buffer)
    indexer_buffer = mem_manager.indexer_ks_mem_manager.kv_buffer

    for layer_idx in range(num_layers):
        for i in range(b_req_idx.shape[0]):
            req_idx = b_req_idx[i].item()
            ready_cache_len = b_ready_cache_len[i].item()
            old_positions = old_indexer_ks_positions[i]

            if ready_cache_len > 0 and old_positions is not None:
                # New positions after copy_kv_index_to_req_prefill
                new_positions = req_to_token_indexs[req_idx, 0:ready_cache_len]

                # Copy indexer_ks: old_positions -> new_positions
                copy_indexer_ks(
                    buffer=indexer_buffer[layer_idx],
                    src_loc=old_positions,
                    dest_loc=new_positions,
                )

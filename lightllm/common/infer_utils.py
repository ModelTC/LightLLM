from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req_prefill


def init_req_to_token_indexes_old(
    req_to_token_indexs, b_req_idx, b_seq_len, b_ready_cache_len, max_len_in_batch, alloc_mem_index
):
    start_index = 0
    b_seq_len_numpy = b_seq_len.cpu().numpy()
    b_ready_cache_len_numpy = b_ready_cache_len.cpu().numpy()
    b_req_idx_numpy = b_req_idx.cpu().numpy()
    for i in range(len(b_seq_len)):
        cur_seq_len = b_seq_len_numpy[i]
        cur_ready_cache_len = b_ready_cache_len_numpy[i]
        req_to_token_indexs[b_req_idx_numpy[i], cur_ready_cache_len:cur_seq_len] = alloc_mem_index[
            start_index : start_index + cur_seq_len - cur_ready_cache_len
        ]
        start_index += cur_seq_len - cur_ready_cache_len
    return


def init_req_to_token_indexes(
    req_to_token_indexs, b_req_idx, b_seq_len, b_ready_cache_len, b_start_loc, alloc_mem_index, max_q_seq_len
):
    copy_kv_index_to_req_prefill(
        req_to_token_indexs=req_to_token_indexs,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        b_ready_cache_len=b_ready_cache_len,
        b_start_loc=b_start_loc,
        memindex=alloc_mem_index,
        max_q_seq_len=max_q_seq_len,
    )

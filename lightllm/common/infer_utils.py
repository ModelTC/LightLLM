from lightllm.backends import get_operator_backend

_operator_backend = get_operator_backend()

copy_kv_index_to_req_prefill = _operator_backend.resolve(
    "common.basemodel.copy_kv_index_to_req.copy_kv_index_to_req_prefill"
)


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

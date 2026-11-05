from lightllm.common.execution.compute.attention.batch import AttentionBatch


def attention_batch(batch):
    return AttentionBatch(
        batch.batch_size,
        batch.max_q_seq_len,
        batch.max_kv_seq_len,
        batch.microbatch_index,
        batch.req_manager.req_to_token_indexs,
        batch.b_req_idx,
        batch.b_seq_len,
        batch.b_q_start_loc,
        batch.b_ready_cache_len,
        batch.b1_cu_q_seq_len,
        batch.b1_cu_kv_seq_len,
        total_kv_tokens=getattr(batch, "total_token_num", None),
        total_q_tokens=batch.input_ids.numel() if getattr(batch, "input_ids", None) is not None else None,
        hold_request_id=getattr(batch.req_manager, "HOLD_REQUEST_ID", None),
        mtp_indices=getattr(batch, "b_mtp_index", None),
        host_seq_lengths=getattr(batch, "host_seq_lengths", None),
        host_cu_q=getattr(batch, "host_cu_q", None),
        host_cu_k=getattr(batch, "host_cu_k", None),
        host_ready=getattr(batch, "host_metadata_ready", None),
    )

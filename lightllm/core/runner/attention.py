from lightllm.core.compute.attention.dense.backend import DenseBatchMetadata


def attention_batch(batch):
    return DenseBatchMetadata(
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


class Runner:
    def __init__(self, plan, decode_phase="decode"):
        self.plan = plan
        self.decode_phase = decode_phase
        self.metadata_sources = {}
        for item in plan.bindings:
            self.metadata_sources.setdefault(item.execution.metadata_key, item.execution)

    def prepare_batch(self, batch):
        phase = "prefill" if batch.is_prefill else self.decode_phase
        executions = [execution for execution in self.metadata_sources.values() if execution.phase == phase]
        if not executions:
            return {}
        inputs = attention_batch(batch)
        return {execution.metadata_key: execution.create_metadata(inputs) for execution in executions}

    def bind_graph(self, metadata, runtime):
        bindings = []
        for key, value in metadata.items():
            bind = getattr(self.metadata_sources[key], "bind_graph", None)
            if bind is not None:
                bindings.append(bind(value, runtime))
        return bindings

"""Construct per-batch state from resources selected during startup."""

from dataclasses import dataclass
from typing import Callable
from lightllm.common.execution.runner.attention import AttentionMetadataRunner

import torch

from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import (
    select_kv_index_from_req,
    select_kv_index_from_req_prefill,
)


def select_token_indices(request_table, batch):
    if batch.is_prefill:
        return select_kv_index_from_req_prefill(
            req_to_token_indexs=request_table,
            b_req_idx=batch.b_req_idx,
            b_seq_len=batch.b_seq_len,
            b_ready_cache_len=batch.b_ready_cache_len,
            b_start_loc=batch.b_prefill_start_loc,
            max_q_seq_len=batch.max_q_seq_len,
            token_num=batch.input_ids.shape[0],
        )
    return select_kv_index_from_req(
        req_to_token_indexs=request_table,
        b_req_idx=batch.b_req_idx,
        b_seq_len=batch.b_seq_len,
    )


@dataclass(frozen=True)
class BatchStateFactory:
    state_type: Callable
    hidden_collector: object
    requests: object
    state_store: object
    role: str
    is_draft: bool
    return_all_prompt_logics: bool
    group_for_lane: Callable
    attention_metadata: AttentionMetadataRunner | None = None
    prefill_states: tuple = ()
    decode_states: tuple = ()
    select_indices: Callable = None
    route_capture: object = None
    prepare_inputs: Callable = None

    def create(self, batch, microbatch_index=0):
        state = self.state_type()
        state.hidden_collector = self.hidden_collector.new_instance()
        state.is_draft_model = self.is_draft
        state.att_state_role = self.role
        state.return_all_prompt_logics = self.return_all_prompt_logics
        state.req_manager = self.requests
        state.att_state_manager = self.state_store
        state.attention_metadata = self.attention_metadata
        state.microbatch_index = microbatch_index
        state.dist_group = self.group_for_lane(microbatch_index)
        assert batch.b_req_idx.shape[0] == batch.b_seq_len.shape[0]
        for name in (
            "input_ids",
            "is_prefill",
            "batch_size",
            "total_token_num",
            "max_q_seq_len",
            "max_kv_seq_len",
            "max_cache_len",
            "b_req_idx",
            "b_seq_len",
            "b_mtp_index",
            "b_position_delta",
            "multimodal_params",
            "mtp_draft_input_hiddens",
        ):
            setattr(state, name, getattr(batch, name))
        if batch.is_prefill:
            state.b_ready_cache_len = (
                batch.b_ready_cache_len if batch.b_ready_cache_len is not None else torch.zeros_like(batch.b_seq_len)
            )
        else:
            state.b_shared_seq_len = batch.b_shared_seq_len
            state.b_shared_radix_node_id = batch.b_shared_radix_node_id
        state.mem_index = (
            select_token_indices(self.requests.req_to_token_indexs, batch)
            if self.select_indices is None
            else self.select_indices(batch)
        )
        state.route_capture = self.route_capture
        if self.prepare_inputs is not None:
            self.prepare_inputs(state)
        if self.route_capture is not None:
            state.moe_capture_callbacks = self.route_capture.bind_batch(state.mem_index)
        factories = self.prefill_states if batch.is_prefill else self.decode_states
        for field, create in factories:
            setattr(state, field, create(infer_state=state))
        return state

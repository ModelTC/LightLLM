from __future__ import annotations

from types import SimpleNamespace

import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo


class Pi0ActionInferStateInfo(InferStateInfo):
    """Block-attention state for the action expert.

    The action suffix is decode-like from a serving perspective, but each
    request contributes a fixed block of queries that attend bidirectionally.
    Selecting prefill attention here keeps that policy local to the action
    model without introducing a second CUDA Graph implementation.
    """

    def __init__(self):
        super().__init__()
        self.position_cos: torch.Tensor | None = None
        self.position_sin: torch.Tensor | None = None
        self.condition: torch.Tensor | None = None
        self.state_infer_state: Pi0ActionInferStateInfo | None = None

    @classmethod
    def create(
        cls,
        model,
        *,
        req_indexes: torch.Tensor,
        prefix_seq_lens: torch.Tensor,
        query_length: int,
        ready_offset: int,
        sequence_offset: int,
    ) -> "Pi0ActionInferStateInfo":
        device = model.mem_manager.req_to_token_indexs.device
        if req_indexes.device != device or prefix_seq_lens.device != device:
            raise ValueError("action runtime metadata must already be on the model device")

        state = cls()
        state.input_ids = torch.empty(
            req_indexes.shape[0] * query_length,
            dtype=torch.int64,
            device=device,
        )
        state.batch_size = req_indexes.shape[0]
        state.total_token_num = state.input_ids.numel()
        state.b_req_idx = req_indexes
        state.b_seq_len = torch.add(prefix_seq_lens, sequence_offset)
        state.b_ready_cache_len = torch.add(prefix_seq_lens, ready_offset)
        state.is_prefill = True
        state.mem_manager = model.mem_manager
        state.req_manager = SimpleNamespace(req_to_token_indexs=model.mem_manager.req_to_token_indexs)
        state.dist_group = model.tp_group
        state.prefill_causal = False
        state.use_ieee_fp32_attention = model.data_type is torch.float32
        state.max_q_seq_len = query_length
        state.max_kv_seq_len = model.max_seq_length
        state.init_some_extra_state(model)
        state.prefill_att_state = model.prefill_att_backend.create_att_prefill_state(state)
        state.init_att_state()
        return state

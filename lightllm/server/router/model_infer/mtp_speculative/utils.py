from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    linear_att_mtp_state_index_update,
    mtp_scatter_next_token_ids,
    mtp_verify,
)
from lightllm.server.router.model_infer.mtp_speculative.planner import SpecDecodePlan

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


def verify_mtp_tokens(
    backend,
    next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    b_mtp_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verify target tokens and update recurrent MTP state when required."""

    accept_lengths, accepted_index = mtp_verify(
        req_to_next_token_ids=backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=next_token_ids,
        b_req_idx=b_req_idx,
    )
    if backend.is_linear_att_mixed_model:
        assert b_mtp_index is not None
        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=backend.model.req_manager.req_to_mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=backend.max_draft_step + 1,
        )
    return accept_lengths, accepted_index


def resolve_mtp_decode_reqs(
    plan: SpecDecodePlan,
    verify_event: torch.cuda.Event,
    run_reqs: List,
    decode_reqs: List,
    accepted_index_cpu: torch.Tensor,
) -> Tuple[List, List]:
    """Return requests ready for pre/post handling after verification."""

    if plan.skip_verify_sync:
        return decode_reqs, decode_reqs

    verify_event.synchronize()
    verify_ok_reqs = [req for req, accepted in zip(run_reqs, accepted_index_cpu.tolist()) if accepted]
    return run_reqs, verify_ok_reqs


def scatter_mtp_next_tokens(
    backend,
    b_req_mtp_start_loc: torch.Tensor,
    all_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    mtp_accept_len: torch.Tensor,
    schedule_scores: Optional[torch.Tensor] = None,
) -> None:
    """Persist the next MTP proposal and optional scheduling scores by request."""

    sampling_params_manager = backend.model.req_manager.req_sampling_params_manager
    mtp_scatter_next_token_ids(
        req_to_next_token_ids=sampling_params_manager.req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        all_next_token_ids=all_next_token_ids,
        b_req_idx=b_req_idx,
        mtp_accept_len=mtp_accept_len,
        req_to_next_token_scores=(
            sampling_params_manager.req_to_next_token_scores if schedule_scores is not None else None
        ),
        schedule_scores=schedule_scores,
    )


def record_request_mtp_metrics(
    backend,
    decode_reqs: List[InferReq],
    accept_lengths_cpu: torch.Tensor,
    verified_row_reqs: List[InferReq],
) -> None:
    """Accumulate user-visible MTP metrics on each request."""

    if not backend.is_master_in_dp:
        return

    accept_lengths = accept_lengths_cpu.tolist()
    assert len(accept_lengths) == len(decode_reqs)
    verify_rows_by_req = Counter(req.req_idx for req in verified_row_reqs)
    for req, accept_len in zip(decode_reqs, accept_lengths):
        req.update_mtp_accepted_token_num(accept_token_num=accept_len - 1)
        verify_token_num = verify_rows_by_req[req.req_idx]
        if verify_token_num > 0:
            req.update_mtp_verify_token_num(verify_token_num=verify_token_num)
            req.update_mtp_verify_step_num(verify_step_num=1)


def free_unused_mtp_decode_mem(
    backend,
    model_input: ModelInput,
    selected_row_mask_cpu: Optional[torch.Tensor],
    accepted_index_cpu: torch.Tensor,
    extra_mem_indexes_cpu: Optional[torch.Tensor],
) -> None:
    """Free rejected target KV slots and draft-only temporary slots."""

    mem_indexes_cpu = model_input.mem_indexes_cpu
    if selected_row_mask_cpu is None:
        free_mask = accepted_index_cpu == 0
    else:
        selected_mask = selected_row_mask_cpu.to(dtype=torch.bool)
        free_mask = selected_mask.logical_not()
        free_mask[selected_mask] = accepted_index_cpu == 0
    need_free_mem_indexes = mem_indexes_cpu[free_mask]

    if extra_mem_indexes_cpu is not None:
        need_free_mem_indexes = torch.cat([need_free_mem_indexes, extra_mem_indexes_cpu], dim=0)
    if len(need_free_mem_indexes) > 0:
        backend.model.req_manager.mem_manager.free(need_free_mem_indexes)


__all__ = [
    "free_unused_mtp_decode_mem",
    "record_request_mtp_metrics",
    "resolve_mtp_decode_reqs",
    "scatter_mtp_next_tokens",
    "verify_mtp_tokens",
]

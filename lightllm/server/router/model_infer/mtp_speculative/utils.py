from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, List, Tuple

import torch

from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    linear_att_mtp_state_index_update,
    mtp_scatter_next_token_ids,
    mtp_verify,
)

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
        MtpMemIndexesToFree,
        SpecProposal,
    )


def alloc_mem_indexes(token_count: int) -> torch.Tensor:
    """Allocate temporary KV slots owned by an MTP proposal."""

    token_count = int(token_count)
    if token_count == 0:
        return torch.empty((0,), dtype=torch.int32, device="cpu")

    from lightllm.server.router.model_infer.infer_batch import g_infer_context

    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(token_count)
    return g_infer_context.req_manager.mem_manager.alloc(token_count)


def verify_mtp_tokens(
    backend: ModeBackend,
    next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    b_mtp_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verify target tokens and update recurrent MTP state when required."""

    accept_lengths, accepted_index = mtp_verify(
        req_to_next_token_ids=backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=next_token_ids,
        b_req_idx=b_req_idx,
    )
    if backend.is_linear_att_mixed_model:
        linear_att_mtp_state_index_update(
            req_to_mtp_state_index=backend.model.req_manager.req_to_mtp_state_index,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            accepted_index=accepted_index,
            verify_width=backend.max_draft_step + 1,
        )
    return accept_lengths, accepted_index


def scatter_mtp_next_tokens(
    backend: ModeBackend,
    proposal: SpecProposal,  # proposal.token_ids: [req_num, draft_step]
    target_next_token_ids: torch.Tensor,  # [verify_batch_size]
    b_req_mtp_start_loc: torch.Tensor,  # [req_num]
    b_req_idx: torch.Tensor,  # [verify_batch_size]
    mtp_accept_len: torch.Tensor,  # [req_num]
) -> None:
    """Persist the next MTP proposal and optional scheduling scores by request."""

    schedule_scores = getattr(proposal, "schedule_scores", None)
    if schedule_scores is not None and 0 in schedule_scores.shape:
        schedule_scores = None

    sampling_params_manager = backend.model.req_manager.req_sampling_params_manager
    mtp_scatter_next_token_ids(
        req_to_next_token_ids=sampling_params_manager.req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        target_next_token_ids=target_next_token_ids,
        draft_token_ids=proposal.token_ids,
        b_req_idx=b_req_idx,
        mtp_accept_len=mtp_accept_len,
        req_to_next_token_scores=(
            sampling_params_manager.req_to_next_token_scores if schedule_scores is not None else None
        ),
        schedule_scores=schedule_scores,
    )


def record_request_mtp_metrics(
    backend: ModeBackend,
    decode_reqs: List[InferReq],
    accept_lengths_cpu: torch.Tensor,
    verify_run_reqs: List[InferReq],
) -> None:
    """Accumulate user-visible MTP metrics on each request."""

    if not backend.is_master_in_dp:
        return

    accept_lengths = accept_lengths_cpu.tolist()
    assert len(accept_lengths) == len(decode_reqs)
    verify_count_by_req_idx = Counter(req.req_idx for req in verify_run_reqs)
    for req, accept_len in zip(decode_reqs, accept_lengths):
        req.update_mtp_accepted_token_num(accept_token_num=accept_len - 1)
        verify_token_num = verify_count_by_req_idx[req.req_idx]
        if verify_token_num > 0:
            req.update_mtp_verify_token_num(verify_token_num=verify_token_num)
            req.update_mtp_verify_step_num(verify_step_num=1)


def free_mem_indexes(
    backend: ModeBackend,
    extra_mem_indexes_cpu: List[MtpMemIndexesToFree],
) -> None:
    """Free all KV indexes described by the unified MTP memory list."""

    mem_indexes_to_free = []
    for extra_mem_to_free in extra_mem_indexes_cpu:
        extra_indexes_cpu = extra_mem_to_free.mem_indexes_cpu
        if extra_mem_to_free.free_mask_cpu is not None:
            extra_indexes_cpu = extra_indexes_cpu[extra_mem_to_free.free_mask_cpu]
        if extra_indexes_cpu.numel() > 0:
            mem_indexes_to_free.append(extra_indexes_cpu)

    if mem_indexes_to_free:
        backend.model.req_manager.mem_manager.free(torch.cat(mem_indexes_to_free, dim=0))


__all__ = [
    "alloc_mem_indexes",
    "free_mem_indexes",
    "record_request_mtp_metrics",
    "scatter_mtp_next_tokens",
    "verify_mtp_tokens",
]

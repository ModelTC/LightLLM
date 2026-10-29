from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, List, Tuple

import torch

from lightllm.common.basemodel.triton_kernel.mtp_utils import (
    mtp_scatter_next_token_ids,
    mtp_verify,
)

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


def verify_mtp_tokens(
    backend: ModeBackend,
    next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    b_mtp_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verify target tokens and update recurrent MTP state when required."""
    # 保留原调用接口中的步序参数；当前状态提交直接使用验证返回的接受长度。

    accept_lengths, accepted_index = mtp_verify(
        req_to_next_token_ids=backend.model.req_manager.req_sampling_params_manager.req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=next_token_ids,
        b_req_idx=b_req_idx,
    )
    backend.model.att_resource_manager.commit_bounded_att_state(
        b_req_idx[b_req_mtp_start_loc], accept_lengths, role="main"
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
    if sampling_params_manager.req_to_bitmask_enabled is not None:
        # 同一流中先写 GPU 主表，再更新常驻 CPU 快照；调用方随后记录的 sync_event 覆盖这次拷贝。
        # 下一轮采样准备等待上一轮 post_handle 交接后才读取，避免拷贝未完成或覆盖仍在读取的快照。
        sampling_params_manager.req_to_next_token_ids_cpu.copy_(
            sampling_params_manager.req_to_next_token_ids, non_blocking=True
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


__all__ = [
    "record_request_mtp_metrics",
    "scatter_mtp_next_tokens",
    "verify_mtp_tokens",
]

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq
    from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
    from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import SpecProposal


def save_dflash2_proposal_state(
    backend: ModeBackend,
    proposal: SpecProposal,
    b_req_idx: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
) -> None:
    """Validate and save a DFlash2 proposal for the next verification step."""

    from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import DFlash2SpecProposal

    if not isinstance(proposal, DFlash2SpecProposal):
        raise TypeError(f"DFlash2 requires DFlash2SpecProposal, got {type(proposal).__name__}")
    request_ids = b_req_idx.index_select(0, b_req_mtp_start_loc.long()).long()
    sampling_manager = backend.model.req_manager.req_sampling_params_manager
    assert sampling_manager.mtp_mode == "dflash2"
    assert request_ids.ndim == 1
    expected_shape = (request_ids.shape[0], *sampling_manager.req_to_dflash2_candidate_ids.shape[1:])
    assert (
        proposal.candidate_ids.shape == proposal.q_probs.shape == expected_shape
    ), "DFlash2 requires a complete fixed-width proposal"
    # 按请求保存候选 token 和分布 q，供下一轮非贪心验证使用。
    sampling_manager.req_to_dflash2_candidate_ids[request_ids] = proposal.candidate_ids
    sampling_manager.req_to_dflash2_q_probs[request_ids] = proposal.q_probs


def _request_uniforms(
    request_reqs: List[InferReq],
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Draw row-wise uniforms while honoring optional per-request generators."""

    uniforms = torch.rand(
        (len(request_reqs), width),
        dtype=torch.float32,
        device=device,
    )
    for row, req in enumerate(request_reqs):
        if req.generator is not None:
            uniforms[row].uniform_(generator=req.generator)
    return uniforms


def sample_and_verify_dflash2_tokens(
    backend: ModeBackend,
    logits: torch.Tensor,
    run_reqs: List[InferReq],
    b_req_idx: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
):
    """Sample and verify a fixed-width DFlash2 block."""

    from lightllm.server.router.model_infer.mode_backend.generic_post_process import build_sampling_probs

    req_num = int(b_req_mtp_start_loc.shape[0])
    verify_width = backend.model.req_manager.req_sampling_params_manager.mtp_verify_width
    assert len(run_reqs) == logits.shape[0]
    assert logits.shape[0] == req_num * verify_width, "DFlash2 requires fixed-width verification"

    sampling_probs, raw_probs = build_sampling_probs(
        logits=logits,
        reqs=run_reqs,
        eos_id=backend.eos_id,
    )
    sampling_manager = backend.model.req_manager.req_sampling_params_manager
    request_ids = b_req_idx.index_select(0, b_req_mtp_start_loc.long()).long()
    proposal_tokens = sampling_manager.req_to_next_token_ids.index_select(0, request_ids)[:, 1:]
    candidate_ids = sampling_manager.req_to_dflash2_candidate_ids.index_select(0, request_ids)
    q_rows = sampling_manager.req_to_dflash2_q_probs.index_select(0, request_ids)

    vocab_size = sampling_probs.shape[-1]
    draft_width = verify_width - 1
    request_reqs = run_reqs[::verify_width]
    return _rejection_sample_from_probs(
        sampling_probs=sampling_probs.view(req_num, verify_width, vocab_size),
        raw_probs=raw_probs.view(req_num, verify_width, vocab_size),
        proposal_tokens=proposal_tokens[:, :draft_width],
        candidate_ids=candidate_ids[:, :draft_width],
        q_rows=q_rows[:, :draft_width],
        request_reqs=request_reqs,
        acceptance_uniforms=_request_uniforms(
            request_reqs,
            draft_width,
            device=sampling_probs.device,
        ),
    )


def _rejection_sample_from_probs(
    sampling_probs: torch.Tensor,
    raw_probs: torch.Tensor,
    proposal_tokens: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    request_reqs: List[InferReq],
    acceptance_uniforms: torch.Tensor | None = None,
):
    """Apply sequential speculative rejection sampling to one DFlash2 block."""

    from lightllm.server.router.model_infer.mode_backend.generic_post_process import (
        _random_sample,
    )

    req_num, verify_width, _ = sampling_probs.shape
    assert proposal_tokens.shape == (req_num, verify_width - 1)
    assert candidate_ids.shape == q_rows.shape
    assert candidate_ids.shape[:2] == proposal_tokens.shape
    assert raw_probs.shape == sampling_probs.shape
    if acceptance_uniforms is not None:
        assert acceptance_uniforms.shape == proposal_tokens.shape

    draft_width = verify_width - 1
    row_ids = torch.arange(req_num, device=sampling_probs.device)
    has_request_seed = any(req.generator is not None for req in request_reqs)

    target_proposal_probs = torch.gather(
        sampling_probs[:, :draft_width],
        dim=-1,
        index=proposal_tokens.unsqueeze(-1),
    ).squeeze(-1)
    draft_proposal_probs = torch.where(
        candidate_ids.eq(proposal_tokens.unsqueeze(-1)),
        q_rows,
        0.0,
    ).sum(dim=-1)
    # 首轮 decode 尚无 proposal，q_rows 为零；必须拒绝占位 token，
    # 让下方的 correction 从完整 target 分布采样。
    accept_probs = torch.where(
        draft_proposal_probs > 0,
        torch.minimum(
            torch.ones_like(target_proposal_probs),
            target_proposal_probs / draft_proposal_probs.clamp_min(1e-20),
        ),
        0.0,
    )
    if acceptance_uniforms is None:
        acceptance_uniforms = torch.rand_like(accept_probs)
    accepted_prefix = acceptance_uniforms.lt(accept_probs).to(torch.int32).cumprod(dim=-1)
    accepted_draft_count = accepted_prefix.sum(dim=-1).to(torch.int32)

    # Only the first rejected position needs a residual sample. If all drafts
    # are accepted, sample the target model's final bonus row instead.
    selected_target_row = accepted_draft_count.long()
    correction_probs = sampling_probs[row_ids, selected_target_row].clone()
    is_rejection = accepted_draft_count.lt(draft_width)
    rejected_slot = accepted_draft_count.clamp_max(draft_width - 1).long()
    rejected_candidate_ids = candidate_ids[row_ids, rejected_slot]
    rejected_q_rows = q_rows[row_ids, rejected_slot] * is_rejection[:, None]
    correction_probs.scatter_add_(1, rejected_candidate_ids, -rejected_q_rows)
    correction_probs.clamp_min_(0.0)
    correction_mass = correction_probs.sum(dim=-1, keepdim=True)
    target_fallback = sampling_probs[row_ids, selected_target_row]
    correction_probs = torch.where(
        correction_mass > 1e-20,
        correction_probs / correction_mass.clamp_min(1e-20),
        target_fallback,
    )
    correction_token = _random_sample(
        correction_probs,
        request_reqs,
        has_request_seed,
    )

    # Rows after the correction token are ignored by accepted_index, but keep
    # them valid so raw-probability gathers and debug tooling are always safe.
    output_ids = torch.zeros((req_num, verify_width), dtype=torch.int64, device=sampling_probs.device)
    output_ids[:, :draft_width] = proposal_tokens
    output_ids.scatter_(1, selected_target_row[:, None], correction_token[:, None])
    accept_lengths = accepted_draft_count + 1
    accepted_index = (torch.arange(verify_width, device=sampling_probs.device)[None, :] < accept_lengths[:, None]).to(
        torch.int32
    )
    flat_output_ids = output_ids.reshape(-1)
    flat_raw_probs = raw_probs.reshape(-1, raw_probs.shape[-1])
    output_logprobs = torch.log(torch.gather(flat_raw_probs, 1, flat_output_ids[:, None]).squeeze(1).clamp_min(1e-20))
    return flat_output_ids, output_logprobs, accept_lengths, accepted_index.reshape(-1)


__all__ = ["sample_and_verify_dflash2_tokens", "save_dflash2_proposal_state"]

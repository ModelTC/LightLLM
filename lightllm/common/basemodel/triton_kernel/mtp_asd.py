"""ASD (Approximate Speculative Decoding, arxiv:2608.03447) acceptance for MTP verify.

This module is a self-contained plugin: it implements the budgeted relaxed acceptance
rule used by ``mtp_speculative.utils.verify_mtp_tokens`` when ``--mtp_asd_regret_budget``
is set. It shares the launch contract of ``mtp_utils.mtp_verify`` so the verification
seam and all downstream consumers stay unchanged.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _fwd_kernel_mtp_asd_verify(
    row_regrets,
    row_draft_token_ids,
    new_next_token_ids,
    req_to_asd_cum_regret,
    mtp_accept_len,
    b_req_mtp_start_loc,
    b_req_idx,
    accepted_index,
    verify_batch_size,
    asd_budget,
    asd_local_ratio,
    asd_max_mismatch,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    req_nums = tl.num_programs(axis=0)

    req_start_loc = tl.load(b_req_mtp_start_loc + cur_index)
    req_start_end = tl.load(
        b_req_mtp_start_loc + cur_index + 1,
        mask=cur_index + 1 < req_nums,
        other=verify_batch_size,
    )
    req_mtp_num = req_start_end - req_start_loc
    draft_num = req_mtp_num - 1  # K draft rows; the final row is the target (bonus) row
    cur_req_idx = tl.load(b_req_idx + req_start_loc)

    offset = tl.arange(0, BLOCK_SIZE)
    req_offset = req_start_loc + offset
    draft_pos_mask = offset < draft_num

    regrets = tl.load(row_regrets + req_offset, mask=draft_pos_mask, other=0.0)
    cum_regrets = tl.cumsum(regrets, axis=0)
    cum_mismatches = tl.cumsum((regrets > 0).to(tl.int32), axis=0)
    suffix_values = draft_num - offset  # K, K-1, ..., 1 on draft rows
    local_ratios = tl.where(suffix_values > 0, regrets / suffix_values, 0.0)

    budget_used = tl.load(req_to_asd_cum_regret + cur_req_idx)
    feasible = (
        draft_pos_mask
        & (budget_used + cum_regrets <= asd_budget)
        & (local_ratios <= asd_local_ratio)
        & (cum_mismatches <= asd_max_mismatch)
    )

    # Acceptance stops at the first infeasible draft position; the target row at that
    # offset is still committed (correction/bonus), mirroring strict accept_len semantics.
    infeasible_positions = tl.where(feasible, BLOCK_SIZE, offset)
    accept_draft_len = tl.min(infeasible_positions, axis=0)
    accept_len = accept_draft_len + 1
    tl.store(mtp_accept_len + cur_index, accept_len)

    accepted_draft_mask = offset < accept_draft_len
    step_regret = tl.sum(tl.where(accepted_draft_mask & draft_pos_mask, regrets, 0.0), axis=0)
    tl.store(req_to_asd_cum_regret + cur_req_idx, budget_used + step_regret)

    # Relaxed rows commit the draft token instead of the target-selected token.
    relaxed_mask = accepted_draft_mask & (regrets > 0)
    draft_token_ids = tl.load(row_draft_token_ids + req_offset, mask=relaxed_mask, other=0)
    tl.store(new_next_token_ids + req_offset, draft_token_ids, mask=relaxed_mask)

    accepted_index_values = tl.where(offset < accept_len, 1, 0)
    tl.store(accepted_index + req_offset, accepted_index_values, mask=offset < req_mtp_num)
    return


def mtp_asd_verify(
    req_to_next_token_ids: torch.Tensor,
    b_req_mtp_start_loc: torch.Tensor,
    new_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    logits: torch.Tensor,
    req_to_asd_cum_regret: torch.Tensor,
    asd_budget: float,
    asd_local_ratio: float,
    asd_max_mismatch: int,
):
    """ASD (Approximate Speculative Decoding, arxiv:2608.03447) variant of ``mtp_verify``.

    A draft token x_i is accepted while (1) the request-level cumulative regret stays within
    ``asd_budget`` (regret r_i = max_v z_i(v) - z_i(x_i) against the target logits z_i),
    (2) r_i / (K - i) <= ``asd_local_ratio``, and (3) the block contains at most
    ``asd_max_mismatch`` relaxed tokens. Acceptance stops at the first infeasible position
    and the target row is committed as usual, so ``asd_budget = 0`` or
    ``asd_max_mismatch = 0`` exactly recovers strict greedy verification.

    Same return contract as ``mtp_verify``; rows accepted under a nonzero regret commit the
    draft token into ``new_next_token_ids`` in place, so downstream consumers (token counter,
    scatter, response building) need no changes.

    Args:
        req_to_next_token_ids: (max_req_num, verify_width)
        b_req_mtp_start_loc: (num_reqs,)
        new_next_token_ids: (verify_batch_size,) target-selected ids; modified in place.
        b_req_idx: (verify_batch_size,)
        b_mtp_index: (verify_batch_size,) local row index of each row within its request.
        logits: (verify_batch_size, vocab) the same logits that produced new_next_token_ids
            (post-penalty, post-temperature).
        req_to_asd_cum_regret: (max_req_num + 1,) per-request cumulative regret ledger.
        asd_budget: request-level cumulative regret budget B.
        asd_local_ratio: per-token regret / suffix-value cap g.
        asd_max_mismatch: max relaxed tokens per verify step m.
    Returns:
        mtp_accept_len: (num_reqs,)
        accepted_index: (verify_batch_size,)
    """
    verify_width = req_to_next_token_ids.shape[1]
    BLOCK_SIZE = 16
    assert verify_width <= BLOCK_SIZE, f"verify_width must be less than {BLOCK_SIZE}"
    num_reqs = b_req_mtp_start_loc.shape[0]
    verify_batch_size = b_req_idx.shape[0]
    assert new_next_token_ids.shape == b_req_idx.shape == b_mtp_index.shape
    assert logits.shape[0] == verify_batch_size

    # Row-level regret precompute (device-side only, no host synchronization).
    # Row r verifies draft column b_mtp_index[r] + 1; the bonus row of each request does
    # not verify any draft, so its column is clamped and its regret is never consumed.
    draft_columns = torch.clamp(b_mtp_index + 1, max=verify_width - 1).to(torch.int64)
    row_draft_token_ids = req_to_next_token_ids[b_req_idx.to(torch.int64), draft_columns]
    row_max_logits = logits.max(dim=-1).values
    row_draft_logits = logits.gather(dim=-1, index=row_draft_token_ids.view(-1, 1)).squeeze(-1)
    row_regrets = (row_max_logits - row_draft_logits).to(torch.float32)

    mtp_accept_len = torch.empty((num_reqs,), dtype=torch.int32, device=req_to_next_token_ids.device)
    accepted_index = torch.empty((verify_batch_size,), dtype=torch.int32, device=req_to_next_token_ids.device)

    grid = (num_reqs,)
    num_warps = 1
    _fwd_kernel_mtp_asd_verify[grid](
        row_regrets=row_regrets,
        row_draft_token_ids=row_draft_token_ids,
        new_next_token_ids=new_next_token_ids,
        req_to_asd_cum_regret=req_to_asd_cum_regret,
        mtp_accept_len=mtp_accept_len,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        b_req_idx=b_req_idx,
        accepted_index=accepted_index,
        verify_batch_size=verify_batch_size,
        asd_budget=asd_budget,
        asd_local_ratio=asd_local_ratio,
        asd_max_mismatch=asd_max_mismatch,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )
    return mtp_accept_len, accepted_index

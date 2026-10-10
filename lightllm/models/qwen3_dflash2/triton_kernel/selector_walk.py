import torch
import triton
import triton.language as tl


@triton.jit
def _greedy_selector_walk_kernel(
    score_lattice,
    candidate_ids,
    draft_token_ids,
    confidence_logits,
    DRAFT_WIDTH: tl.constexpr,
    TOP_K: tl.constexpr,
    WRITE_CONFIDENCE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    candidate_indices = tl.arange(0, TOP_K)
    # The first position is conditioned on the anchor, so any predecessor row works.
    predecessor_candidate_idx = 0
    for draft_idx in range(DRAFT_WIDTH):
        conditional_score_offset = ((req_idx * DRAFT_WIDTH + draft_idx) * TOP_K + predecessor_candidate_idx) * TOP_K
        conditional_scores = tl.load(score_lattice + conditional_score_offset + candidate_indices).to(tl.float32)
        max_score = tl.max(conditional_scores, axis=0)
        selected_candidate_idx = tl.min(tl.where(conditional_scores == max_score, candidate_indices, TOP_K), axis=0)
        draft_token_offset = req_idx * DRAFT_WIDTH + draft_idx
        draft_token_id = tl.load(candidate_ids + draft_token_offset * TOP_K + selected_candidate_idx)
        tl.store(draft_token_ids + draft_token_offset, draft_token_id)
        if WRITE_CONFIDENCE:
            # Log-odds of the selected candidate within this conditional top-k row.
            # Its sigmoid is a scheduling heuristic, not the greedy proposal's q.
            other_candidate_mass = tl.sum(
                tl.where(candidate_indices == selected_candidate_idx, 0.0, tl.exp(conditional_scores - max_score)),
                axis=0,
            )
            tl.store(confidence_logits + draft_token_offset, -tl.log(other_candidate_mass))
        predecessor_candidate_idx = selected_candidate_idx


@torch.no_grad()
def greedy_selector_walk(
    score_lattice: torch.Tensor,
    candidate_ids: torch.Tensor,
    confidence_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select draft_token_ids [req_num, draft_width] from the conditional scores.

    score_lattice: [req_num, draft_width, predecessor_top_k, successor_top_k].
        At draft_idx=0, each predecessor row repeats the anchor-conditioned scores.
    candidate_ids: [req_num, draft_width, top_k], containing vocabulary token IDs.
    confidence_logits: optional [req_num, draft_width] output of conditional log-odds.
    """
    req_num, draft_width, top_k, successor_top_k = score_lattice.shape
    assert top_k == successor_top_k == triton.next_power_of_2(top_k)
    assert candidate_ids.shape == (req_num, draft_width, top_k)
    if confidence_logits is not None:
        assert confidence_logits.shape == (req_num, draft_width) and confidence_logits.is_contiguous()
        assert confidence_logits.dtype == torch.float32 and confidence_logits.device == score_lattice.device
    draft_token_ids = torch.empty((req_num, draft_width), dtype=torch.int64, device=score_lattice.device)
    if req_num:
        _greedy_selector_walk_kernel[(req_num,)](
            score_lattice.contiguous(),
            candidate_ids.contiguous(),
            draft_token_ids,
            confidence_logits,
            DRAFT_WIDTH=draft_width,
            TOP_K=top_k,
            WRITE_CONFIDENCE=confidence_logits is not None,
            num_warps=1,
            num_stages=1,
        )
    return draft_token_ids

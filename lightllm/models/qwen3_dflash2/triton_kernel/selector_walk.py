import torch
import triton
import triton.language as tl


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ids_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_mask_ptr,
    tokens_ptr,
    q_ptr,
    path_indices_ptr,
    SLOT_NUM: tl.constexpr,
    TOP_K: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offsets = tl.arange(0, TOP_K)
    temperature = tl.load(temperatures_ptr + req_idx)
    is_greedy = tl.load(greedy_mask_ptr + req_idx) != 0
    previous_index = 0

    for slot_idx in range(SLOT_NUM):
        score_offset = ((req_idx * SLOT_NUM + slot_idx) * TOP_K + previous_index) * TOP_K
        scores = tl.load(scores_ptr + score_offset + offsets).to(tl.float32)

        if is_greedy:
            best_score = tl.max(scores, axis=0)
            selected_index = tl.min(tl.where(scores == best_score, offsets, TOP_K), axis=0)
            probabilities = tl.where(offsets == selected_index, 1.0, 0.0)
        else:
            scaled_scores = scores / temperature
            exponentials = tl.exp(scaled_scores - tl.max(scaled_scores, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + req_idx * SLOT_NUM + slot_idx)
            selected_index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0),
                axis=0,
            )
            selected_index = tl.minimum(selected_index, TOP_K - 1)

        output_offset = req_idx * SLOT_NUM + slot_idx
        candidate_offset = output_offset * TOP_K
        tl.store(q_ptr + candidate_offset + offsets, probabilities)
        tl.store(tokens_ptr + output_offset, tl.load(candidate_ids_ptr + candidate_offset + selected_index))
        tl.store(path_indices_ptr + output_offset, selected_index)
        previous_index = selected_index


@torch.no_grad()
def selector_walk(
    scores: torch.Tensor,
    candidate_ids: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
):
    """Sample one locally coherent candidate path and retain every conditional q row."""

    req_num, slot_num, top_k, successor_top_k = scores.shape
    assert top_k == successor_top_k
    assert candidate_ids.shape == (req_num, slot_num, top_k)
    assert uniforms.shape == (req_num, slot_num)
    assert temperatures.shape == (req_num,)
    assert greedy_mask.shape == (req_num,)
    assert top_k == triton.next_power_of_2(top_k)

    scores = scores.contiguous()
    candidate_ids = candidate_ids.contiguous()
    uniforms = uniforms.contiguous()
    temperatures = temperatures.contiguous()
    greedy_mask = greedy_mask.contiguous()
    tokens = torch.empty((req_num, slot_num), dtype=torch.int64, device=scores.device)
    q_rows = torch.empty((req_num, slot_num, top_k), dtype=torch.float32, device=scores.device)
    path_indices = torch.empty((req_num, slot_num), dtype=torch.int64, device=scores.device)

    _selector_walk_kernel[(req_num,)](
        scores,
        candidate_ids,
        uniforms,
        temperatures,
        greedy_mask,
        tokens,
        q_rows,
        path_indices,
        SLOT_NUM=slot_num,
        TOP_K=top_k,
        num_warps=1,
        num_stages=1,
    )
    return tokens, q_rows, path_indices

import triton
import triton.language as tl
import torch


@triton.jit
def _fwd_kernel_cumprod_scores(
    req_to_next_token_scores,
    req_to_next_token_scores_stride,
    b_req_idx,
    max_draft_step,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index * (max_draft_step + 1))
    base_ptr = req_to_next_token_scores + cur_req_idx * req_to_next_token_scores_stride
    tl.store(base_ptr, 1.0)

    offset = tl.arange(0, BLOCK_SIZE)
    store_mask = offset < (max_draft_step + 1)

    scores = tl.load(base_ptr + offset, mask=store_mask, other=0.0)
    # offset 0 是 target sample，本轮恒接受；只有 draft 条件接受概率需要 clamp。
    scores = tl.where(offset == 0, 1.0, scores)
    # Clamp draft scheduling scores before converting them to prefix-survival scores.
    # This makes each request's cumulative acceptance probabilities monotonic,
    # so global top-k selection cannot pick a later draft row without its prefix.
    scores = tl.where((offset != 0) & (scores >= 0.99), 0.99, scores)
    scores = tl.where((offset != 0) & (scores <= 0.01), 0.01, scores)

    cumulative_scores = tl.cumprod(scores, axis=0)

    tl.store(base_ptr + offset, cumulative_scores, mask=store_mask)
    return


def sample_dynamic_spec_row_mask(
    dynamic_batch_size: int,
    b_req_idx: torch.Tensor,
    req_to_next_token_scores: torch.Tensor,
    max_draft_step: int,
    pre_draft_step: int = None,
) -> torch.Tensor:
    dynamic_batch_size = int(dynamic_batch_size)
    max_draft_step = int(max_draft_step)
    pre_draft_step = max_draft_step if pre_draft_step is None else int(pre_draft_step)
    assert 0 <= pre_draft_step <= max_draft_step
    assert b_req_idx.shape[0] % (max_draft_step + 1) == 0
    assert req_to_next_token_scores.is_cuda
    assert dynamic_batch_size <= b_req_idx.shape[0]
    req_num = len(b_req_idx) // (max_draft_step + 1)
    valid_row_num = req_num * (pre_draft_step + 1)
    assert dynamic_batch_size <= valid_row_num

    # Convert each request's conditional scheduling scores to prefix survival scores.
    _fwd_kernel_cumprod_scores[(req_num,)](
        req_to_next_token_scores=req_to_next_token_scores,
        req_to_next_token_scores_stride=req_to_next_token_scores.stride(0),
        b_req_idx=b_req_idx,
        max_draft_step=max_draft_step,
        BLOCK_SIZE=triton.next_power_of_2(max_draft_step + 1),
        num_warps=1,
        num_stages=1,
    )

    request_ids = b_req_idx[:: max_draft_step + 1].long()
    scores = req_to_next_token_scores.index_select(0, request_ids)[:, : pre_draft_step + 1].flatten()
    compact_ids = torch.topk(scores, k=dynamic_batch_size, sorted=False).indices
    request_offsets = compact_ids // (pre_draft_step + 1)
    step_offsets = compact_ids % (pre_draft_step + 1)
    selected_ids = request_offsets * (max_draft_step + 1) + step_offsets

    selected_row_mask = torch.zeros((len(b_req_idx),), dtype=torch.int32, device=b_req_idx.device)
    selected_row_mask.scatter_(0, selected_ids, 1)
    return selected_row_mask

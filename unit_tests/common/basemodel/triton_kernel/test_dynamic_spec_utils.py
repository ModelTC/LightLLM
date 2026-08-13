import torch
import pytest
import triton
import numpy as np

from lightllm.common.basemodel.triton_kernel.dynamic_spec_utils import (
    _fwd_kernel_cumprod_scores,
    sample_dynamic_spec_row_mask,
)


def _reference_cumprod_scores(req_to_next_token_scores, b_req_idx, max_draft_step: int) -> torch.Tensor:
    scores = req_to_next_token_scores.clone()
    req_num = b_req_idx.shape[0] // (max_draft_step + 1)
    for req_i in range(req_num):
        req_idx = int(b_req_idx[req_i * (max_draft_step + 1)].item())
        row = scores[req_idx, : max_draft_step + 1].clone()
        row[0] = 1.0
        row[1:] = torch.clamp(row[1:], min=0.01, max=0.99)
        scores[req_idx, : max_draft_step + 1] = torch.cumprod(row, dim=0)
    return scores


def _flat_cumprod_scores(
    b_req_idx: torch.Tensor,
    req_to_next_token_scores: torch.Tensor,
    max_draft_step: int,
) -> torch.Tensor:
    scores = _reference_cumprod_scores(req_to_next_token_scores, b_req_idx, max_draft_step)
    req_num = b_req_idx.shape[0] // (max_draft_step + 1)
    all_num = req_num * (max_draft_step + 1)
    flat_scores = []
    for offset in range(all_num):
        req_idx = int(b_req_idx[offset].item())
        mtp_index = offset % (max_draft_step + 1)
        flat_scores.append(scores[req_idx, mtp_index])
    return torch.stack(flat_scores)


def _assert_topk_mask(select: torch.Tensor, flat_scores: torch.Tensor, dynamic_batch_size: int) -> None:
    k = dynamic_batch_size
    assert int(select.sum().item()) == k
    selected_scores = flat_scores[select.bool()]
    unselected_scores = flat_scores[(select == 0).bool()]
    if unselected_scores.numel() > 0:
        assert selected_scores.min() >= unselected_scores.max() - 1e-5


def _make_batch_scores(req_num: int, max_draft_step: int, rows):
    max_req = req_num
    scores = torch.zeros((max_req + 1, 16), dtype=torch.float32, device="cuda")
    for req_idx, row in enumerate(rows):
        scores[req_idx, : max_draft_step + 1] = torch.tensor(row, dtype=torch.float32, device="cuda")
    b_req_idx = torch.arange(req_num, dtype=torch.int32, device="cuda").repeat_interleave(max_draft_step + 1)
    return scores, b_req_idx


@pytest.mark.parametrize("max_draft_step", [1, 3])
def test_cumprod_scores_kernel(max_draft_step: int):
    req_num = 2
    scores, b_req_idx = _make_batch_scores(
        req_num,
        max_draft_step,
        rows=[
            [1.0] + [0.5] * max_draft_step,
            [1.0] + [0.2] * max_draft_step,
        ],
    )
    scores_clone = scores.clone()
    _fwd_kernel_cumprod_scores[(req_num,)](
        req_to_next_token_scores=scores_clone,
        req_to_next_token_scores_stride=scores_clone.stride(0),
        b_req_idx=b_req_idx,
        max_draft_step=max_draft_step,
        BLOCK_SIZE=triton.next_power_of_2(max_draft_step + 1),
        num_warps=1,
        num_stages=1,
    )
    expected = _reference_cumprod_scores(scores, b_req_idx, max_draft_step)
    assert torch.allclose(
        scores_clone[:, : max_draft_step + 1], expected[:, : max_draft_step + 1], rtol=1e-5, atol=1e-5
    )


def test_cumprod_scores_clamps_invalid_values():
    max_draft_step = 2
    req_num = 1
    scores, b_req_idx = _make_batch_scores(req_num, max_draft_step, rows=[[1.0, 0.0, 1.5]])
    _fwd_kernel_cumprod_scores[(req_num,)](
        req_to_next_token_scores=scores,
        req_to_next_token_scores_stride=scores.stride(0),
        b_req_idx=b_req_idx,
        max_draft_step=max_draft_step,
        BLOCK_SIZE=triton.next_power_of_2(max_draft_step + 1),
        num_warps=1,
        num_stages=1,
    )
    row = scores[0, : max_draft_step + 1]
    # Index 0 is the always-accepted target sample; clamping applies only to drafts.
    assert row[0].item() == pytest.approx(1.0)
    assert row[1].item() == pytest.approx(0.01, rel=1e-4)
    assert row[2].item() == pytest.approx(0.01 * 0.99, rel=1e-4)


def test_cumprod_scores_clamps_boundary_values():
    max_draft_step = 3
    req_num = 1
    scores, b_req_idx = _make_batch_scores(req_num, max_draft_step, rows=[[1.0, 0.995, 0.005, 0.5]])
    raw_scores = scores.clone()
    _fwd_kernel_cumprod_scores[(req_num,)](
        req_to_next_token_scores=scores,
        req_to_next_token_scores_stride=scores.stride(0),
        b_req_idx=b_req_idx,
        max_draft_step=max_draft_step,
        BLOCK_SIZE=triton.next_power_of_2(max_draft_step + 1),
        num_warps=1,
        num_stages=1,
    )
    expected = _reference_cumprod_scores(raw_scores, b_req_idx, max_draft_step)
    row = scores[0, : max_draft_step + 1]
    assert torch.allclose(row, expected[0, : max_draft_step + 1], rtol=1e-5, atol=1e-5)
    # Draft scores 0.995 and 0.005 clamp to 0.99 and 0.01.
    assert row[0].item() == pytest.approx(1.0)
    assert row[1].item() == pytest.approx(0.99, rel=1e-4)
    assert row[2].item() == pytest.approx(0.99 * 0.01, rel=1e-4)
    assert row[3].item() == pytest.approx(0.99 * 0.01 * 0.5, rel=1e-4)


def test_sample_select_count():
    max_draft_step = 3
    req_num = 3
    scores, b_req_idx = _make_batch_scores(
        req_num,
        max_draft_step,
        rows=[
            [1.0, 0.95, 0.90, 0.10],
            [1.0, 0.20, 0.80, 0.80],
            [1.0, 0.99, 0.99, 0.99],
        ],
    )
    all_num = req_num * (max_draft_step + 1)
    for dynamic_batch_size in [3, 8, all_num]:
        select = sample_dynamic_spec_row_mask(
            dynamic_batch_size=dynamic_batch_size,
            b_req_idx=b_req_idx,
            req_to_next_token_scores=scores.clone(),
            max_draft_step=max_draft_step,
        )
        assert select.dtype == torch.int32
        assert select.shape[0] == all_num
        assert int(select.sum().item()) == dynamic_batch_size
        assert torch.all((select == 0) | (select == 1))


def test_sample_accepts_numpy_scalar_dynamic_batch_size():
    max_draft_step = 3
    scores, b_req_idx = _make_batch_scores(
        3,
        max_draft_step,
        rows=[
            [1.0, 0.95, 0.90, 0.10],
            [1.0, 0.20, 0.80, 0.80],
            [1.0, 0.99, 0.99, 0.99],
        ],
    )
    select = sample_dynamic_spec_row_mask(
        dynamic_batch_size=np.int64(8),
        b_req_idx=b_req_idx,
        req_to_next_token_scores=scores,
        max_draft_step=np.int64(max_draft_step),
    )
    assert int(select.sum().item()) == 8


def test_sample_topk_by_cumprod_score():
    max_draft_step = 3
    scores, b_req_idx = _make_batch_scores(
        3,
        max_draft_step,
        rows=[
            [1.0, 0.95, 0.90, 0.10],
            [1.0, 0.20, 0.80, 0.80],
            [1.0, 0.99, 0.99, 0.99],
        ],
    )
    flat_scores = _flat_cumprod_scores(b_req_idx, scores, max_draft_step)
    for dynamic_batch_size in [1, 4, 8, 12]:
        select = sample_dynamic_spec_row_mask(
            dynamic_batch_size=dynamic_batch_size,
            b_req_idx=b_req_idx,
            req_to_next_token_scores=scores.clone(),
            max_draft_step=max_draft_step,
        )
        _assert_topk_mask(select, flat_scores, dynamic_batch_size)


def test_sample_picks_highest_cumprod_rows():
    max_draft_step = 1
    scores, b_req_idx = _make_batch_scores(
        2,
        max_draft_step,
        rows=[
            [1.0, 0.9],
            [1.0, 0.1],
        ],
    )
    flat_scores = _flat_cumprod_scores(b_req_idx, scores, max_draft_step)
    select = sample_dynamic_spec_row_mask(
        dynamic_batch_size=2,
        b_req_idx=b_req_idx,
        req_to_next_token_scores=scores.clone(),
        max_draft_step=max_draft_step,
    )
    _assert_topk_mask(select, flat_scores, 2)
    # top-2 scores are both 0.99 at mtp_index==0 (req0 and req1 main rows)
    assert select[0].item() == 1
    assert select[2].item() == 1


def test_sample_single_request():
    max_draft_step = 2
    scores, b_req_idx = _make_batch_scores(1, max_draft_step, rows=[[1.0, 0.5, 0.25]])
    flat_scores = _flat_cumprod_scores(b_req_idx, scores, max_draft_step)
    select = sample_dynamic_spec_row_mask(
        dynamic_batch_size=2,
        b_req_idx=b_req_idx,
        req_to_next_token_scores=scores.clone(),
        max_draft_step=max_draft_step,
    )
    _assert_topk_mask(select, flat_scores, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

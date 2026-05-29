import torch
import pytest
import triton
import numpy as np

from lightllm.common.basemodel.triton_kernel.mtp_utils1 import (
    _fwd_kernel_cumprod_probs,
    sample_dynamic_mtp_req_mask,
)


def _reference_cumprod_probs(req_to_next_token_probs, b_req_idx, mtp_step: int) -> torch.Tensor:
    probs = req_to_next_token_probs.clone()
    req_num = b_req_idx.shape[0] // (mtp_step + 1)
    for req_i in range(req_num):
        req_idx = int(b_req_idx[req_i * (mtp_step + 1)].item())
        row = probs[req_idx, : mtp_step + 1].clone()
        row[0] = 1.0
        row = torch.where(row >= 0.99, torch.tensor(0.99, device=row.device, dtype=row.dtype), row)
        row = torch.where(row <= 0.01, torch.tensor(0.01, device=row.device, dtype=row.dtype), row)
        probs[req_idx, : mtp_step + 1] = torch.cumprod(row, dim=0)
    return probs


def _flat_cumprod_probs(
    b_req_idx: torch.Tensor,
    req_to_next_token_probs: torch.Tensor,
    mtp_step: int,
) -> torch.Tensor:
    probs = _reference_cumprod_probs(req_to_next_token_probs, b_req_idx, mtp_step)
    req_num = b_req_idx.shape[0] // (mtp_step + 1)
    all_num = req_num * (mtp_step + 1)
    flat_probs = []
    for offset in range(all_num):
        req_idx = int(b_req_idx[offset].item())
        mtp_index = offset % (mtp_step + 1)
        flat_probs.append(probs[req_idx, mtp_index])
    return torch.stack(flat_probs)


def _assert_topk_mask(select: torch.Tensor, flat_probs: torch.Tensor, dynamic_batch_size: int) -> None:
    k = dynamic_batch_size
    assert int(select.sum().item()) == k
    selected_scores = flat_probs[select.bool()]
    unselected_scores = flat_probs[(select == 0).bool()]
    if unselected_scores.numel() > 0:
        assert selected_scores.min() >= unselected_scores.max() - 1e-5


def _make_batch_probs(req_num: int, mtp_step: int, rows):
    max_req = req_num
    probs = torch.zeros((max_req + 1, 16), dtype=torch.float32, device="cuda")
    for req_idx, row in enumerate(rows):
        probs[req_idx, : mtp_step + 1] = torch.tensor(row, dtype=torch.float32, device="cuda")
    b_req_idx = torch.arange(req_num, dtype=torch.int32, device="cuda").repeat_interleave(mtp_step + 1)
    return probs, b_req_idx


@pytest.mark.parametrize("mtp_step", [1, 3])
def test_cumprod_probs_kernel(mtp_step: int):
    req_num = 2
    probs, b_req_idx = _make_batch_probs(
        req_num,
        mtp_step,
        rows=[
            [1.0] + [0.5] * mtp_step,
            [1.0] + [0.2] * mtp_step,
        ],
    )
    probs_clone = probs.clone()
    _fwd_kernel_cumprod_probs[(req_num,)](
        req_to_next_token_probs=probs_clone,
        req_to_next_token_probs_stride=probs_clone.stride(0),
        b_req_idx=b_req_idx,
        mtp_step=mtp_step,
        BLOCK_SIZE=triton.next_power_of_2(mtp_step + 1),
        num_warps=1,
        num_stages=1,
    )
    expected = _reference_cumprod_probs(probs, b_req_idx, mtp_step)
    assert torch.allclose(probs_clone[:, : mtp_step + 1], expected[:, : mtp_step + 1], rtol=1e-5, atol=1e-5)


def test_cumprod_probs_clamps_invalid_values():
    mtp_step = 2
    req_num = 1
    probs, b_req_idx = _make_batch_probs(req_num, mtp_step, rows=[[1.0, 0.0, 1.5]])
    _fwd_kernel_cumprod_probs[(req_num,)](
        req_to_next_token_probs=probs,
        req_to_next_token_probs_stride=probs.stride(0),
        b_req_idx=b_req_idx,
        mtp_step=mtp_step,
        BLOCK_SIZE=triton.next_power_of_2(mtp_step + 1),
        num_warps=1,
        num_stages=1,
    )
    row = probs[0, : mtp_step + 1]
    # index 0 is written as 1.0, then clamped (>=0.99 -> 0.99, <=0.01 -> 0.01) before cumprod
    assert row[0].item() == pytest.approx(0.99)
    assert row[1].item() == pytest.approx(0.99 * 0.01, rel=1e-4)
    assert row[2].item() == pytest.approx(0.99 * 0.01 * 0.99, rel=1e-4)


def test_cumprod_probs_clamps_boundary_values():
    mtp_step = 3
    req_num = 1
    probs, b_req_idx = _make_batch_probs(req_num, mtp_step, rows=[[1.0, 0.995, 0.005, 0.5]])
    raw_probs = probs.clone()
    _fwd_kernel_cumprod_probs[(req_num,)](
        req_to_next_token_probs=probs,
        req_to_next_token_probs_stride=probs.stride(0),
        b_req_idx=b_req_idx,
        mtp_step=mtp_step,
        BLOCK_SIZE=triton.next_power_of_2(mtp_step + 1),
        num_warps=1,
        num_stages=1,
    )
    expected = _reference_cumprod_probs(raw_probs, b_req_idx, mtp_step)
    row = probs[0, : mtp_step + 1]
    assert torch.allclose(row, expected[0, : mtp_step + 1], rtol=1e-5, atol=1e-5)
    # 0.995 -> 0.99, 0.005 -> 0.01, then cumprod
    assert row[0].item() == pytest.approx(0.99)
    assert row[1].item() == pytest.approx(0.99 * 0.99, rel=1e-4)
    assert row[2].item() == pytest.approx(0.99 * 0.99 * 0.01, rel=1e-4)
    assert row[3].item() == pytest.approx(0.99 * 0.99 * 0.01 * 0.5, rel=1e-4)


def test_sample_select_count():
    mtp_step = 3
    req_num = 3
    probs, b_req_idx = _make_batch_probs(
        req_num,
        mtp_step,
        rows=[
            [1.0, 0.95, 0.90, 0.10],
            [1.0, 0.20, 0.80, 0.80],
            [1.0, 0.99, 0.99, 0.99],
        ],
    )
    all_num = req_num * (mtp_step + 1)
    for dynamic_batch_size in [3, 8, all_num]:
        select = sample_dynamic_mtp_req_mask(
            dynamic_batch_size=dynamic_batch_size,
            b_req_idx=b_req_idx,
            req_to_next_token_probs=probs.clone(),
            mtp_step=mtp_step,
        )
        assert select.dtype == torch.int32
        assert select.shape[0] == all_num
        assert int(select.sum().item()) == dynamic_batch_size
        assert torch.all((select == 0) | (select == 1))


def test_sample_accepts_numpy_scalar_dynamic_batch_size():
    mtp_step = 3
    probs, b_req_idx = _make_batch_probs(
        3,
        mtp_step,
        rows=[
            [1.0, 0.95, 0.90, 0.10],
            [1.0, 0.20, 0.80, 0.80],
            [1.0, 0.99, 0.99, 0.99],
        ],
    )
    select = sample_dynamic_mtp_req_mask(
        dynamic_batch_size=np.int64(8),
        b_req_idx=b_req_idx,
        req_to_next_token_probs=probs,
        mtp_step=np.int64(mtp_step),
    )
    assert int(select.sum().item()) == 8


def test_sample_topk_by_cumprod_score():
    mtp_step = 3
    probs, b_req_idx = _make_batch_probs(
        3,
        mtp_step,
        rows=[
            [1.0, 0.95, 0.90, 0.10],
            [1.0, 0.20, 0.80, 0.80],
            [1.0, 0.99, 0.99, 0.99],
        ],
    )
    flat_probs = _flat_cumprod_probs(b_req_idx, probs, mtp_step)
    for dynamic_batch_size in [1, 4, 8, 12]:
        select = sample_dynamic_mtp_req_mask(
            dynamic_batch_size=dynamic_batch_size,
            b_req_idx=b_req_idx,
            req_to_next_token_probs=probs.clone(),
            mtp_step=mtp_step,
        )
        _assert_topk_mask(select, flat_probs, dynamic_batch_size)


def test_sample_picks_highest_cumprod_rows():
    mtp_step = 1
    probs, b_req_idx = _make_batch_probs(
        2,
        mtp_step,
        rows=[
            [1.0, 0.9],
            [1.0, 0.1],
        ],
    )
    flat_probs = _flat_cumprod_probs(b_req_idx, probs, mtp_step)
    select = sample_dynamic_mtp_req_mask(
        dynamic_batch_size=2,
        b_req_idx=b_req_idx,
        req_to_next_token_probs=probs.clone(),
        mtp_step=mtp_step,
    )
    _assert_topk_mask(select, flat_probs, 2)
    # top-2 scores are both 0.99 at mtp_index==0 (req0 and req1 main rows)
    assert select[0].item() == 1
    assert select[2].item() == 1


def test_sample_single_request():
    mtp_step = 2
    probs, b_req_idx = _make_batch_probs(1, mtp_step, rows=[[1.0, 0.5, 0.25]])
    flat_probs = _flat_cumprod_probs(b_req_idx, probs, mtp_step)
    select = sample_dynamic_mtp_req_mask(
        dynamic_batch_size=2,
        b_req_idx=b_req_idx,
        req_to_next_token_probs=probs.clone(),
        mtp_step=mtp_step,
    )
    _assert_topk_mask(select, flat_probs, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

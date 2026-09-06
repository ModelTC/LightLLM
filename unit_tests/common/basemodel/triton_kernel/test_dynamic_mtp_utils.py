import json

import torch
import pytest
import triton
import numpy as np

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.mtp_manager import MtpManager
from lightllm.common.basemodel.triton_kernel import dynamic_mtp_utils
from lightllm.common.basemodel.triton_kernel.dynamic_mtp_utils import (
    _fwd_kernel_cumprod_scores,
    sample_dynamic_mtp_row_mask,
)
from lightllm.utils.envs_utils import get_env_start_args


@pytest.fixture(autouse=True)
def _reset_mtp_manager():
    MtpManager._instance = None
    yield
    MtpManager._instance = None


def test_compact_dynamic_mtp_model_input(monkeypatch):
    monkeypatch.setenv(
        "LIGHTLLM_START_ARGS",
        json.dumps(
            {
                "diverse_mode": False,
                "llm_kv_type": "fp16",
                "mtp_mode": "eagle3",
                "mtp_dynamic_verify": True,
                "mtp_step": 3,
                "llm_decode_att_backend": "triton",
            }
        ),
    )
    monkeypatch.setenv("LIGHTLLM_MAX_BATCH_SHARED_GROUP_SIZE", "4")
    get_env_start_args.cache_clear()

    model_input = ModelInput(
        batch_size=12,
        total_token_num=54,
        max_q_seq_len=1,
        max_kv_seq_len=6,
        input_ids=torch.arange(12, dtype=torch.int64, device="cuda") + 1000,
        b_req_idx=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.int32, device="cuda"),
        b_mtp_index=torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6], dtype=torch.int32, device="cuda"),
        b_position_delta=torch.arange(12, dtype=torch.int32, device="cuda") + 200,
        b_shared_seq_len=torch.tensor([0, 0, 0, 0, 7, 7, 7, 7, 9, 9, 9, 9], dtype=torch.int32, device="cuda"),
        b_shared_radix_node_id=torch.tensor(
            [10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12], dtype=torch.int64, device="cuda"
        ),
        mem_indexes=torch.arange(8, dtype=torch.int32, device="cuda") + 100,
        mem_indexes_cpu=torch.arange(8, dtype=torch.int32, device="cpu") + 100,
        is_prefill=False,
        multimodal_params=[{"row": i, "images": [], "audios": []} for i in range(12)],
        mtp_draft_input_hiddens=(torch.arange(12 * 5, dtype=torch.float32, device="cuda").reshape(12, 5) + 0.5),
    )
    req_to_next_token_scores = torch.tensor(
        [
            [1.0, 0.95, 0.90, 0.10, 0.0, 0.0],
            [1.0, 0.20, 0.80, 0.80, 0.0, 0.0],
            [1.0, 0.99, 0.99, 0.99, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    compacted_input, selected_row_mask = dynamic_mtp_utils.prepare_dynamic_mtp_model_input(
        model_input=model_input,
        req_num=3,
        dynamic_batch_size=8,
        req_to_next_token_scores=req_to_next_token_scores,
    )
    torch.cuda.synchronize()

    expected_selected_mask = torch.tensor([1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    expected_selected_rows = torch.where(expected_selected_mask == 1)[0]

    assert torch.equal(selected_row_mask.cpu(), expected_selected_mask)
    assert compacted_input.batch_size == 8
    assert compacted_input.max_q_seq_len == 1
    assert compacted_input.multimodal_params == [{"images": [], "audios": []}] * 8

    assert torch.equal(
        compacted_input.input_ids.cpu(), torch.arange(12, dtype=torch.int64)[expected_selected_rows] + 1000
    )
    assert torch.equal(compacted_input.b_req_idx.cpu(), torch.tensor([0, 0, 0, 1, 2, 2, 2, 2], dtype=torch.int32))
    assert torch.equal(compacted_input.b_mtp_index.cpu(), torch.tensor([0, 1, 2, 0, 0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(compacted_input.b_seq_len.cpu(), torch.tensor([3, 4, 5, 3, 3, 4, 5, 6], dtype=torch.int32))
    assert torch.equal(
        compacted_input.b_position_delta.cpu(),
        torch.tensor([200, 201, 202, 204, 208, 209, 210, 211], dtype=torch.int32),
    )
    assert torch.equal(
        compacted_input.b_shared_seq_len.cpu(), torch.tensor([0, 0, 0, 7, 9, 9, 9, 9], dtype=torch.int32)
    )
    assert torch.equal(
        compacted_input.b_shared_radix_node_id.cpu(),
        torch.tensor([10, 10, 10, 11, 12, 12, 12, 12], dtype=torch.int64),
    )
    assert torch.equal(
        compacted_input.mem_indexes.cpu(), torch.tensor([100, 101, 102, 103, 104, 105, 106, 107], dtype=torch.int32)
    )
    # CPU/GPU mem indexes are trimmed by SpecEngine.prepare_decode_model_input
    # before entering this lower-level row compaction helper.
    assert torch.equal(compacted_input.mem_indexes_cpu, torch.arange(8, dtype=torch.int32) + 100)

    expected_hiddens = (torch.arange(12 * 5, dtype=torch.float32).reshape(12, 5) + 0.5)[expected_selected_rows]
    assert torch.equal(compacted_input.mtp_draft_input_hiddens.cpu(), expected_hiddens)


def test_compaction_preserves_shared_radix_metadata():
    model_input = ModelInput(
        batch_size=5,
        total_token_num=36,
        max_q_seq_len=1,
        max_kv_seq_len=6,
        input_ids=None,
        b_req_idx=torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"),
        b_mtp_index=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([3, 4, 5, 6, 7], dtype=torch.int32, device="cuda"),
        b_position_delta=torch.zeros(5, dtype=torch.int32, device="cuda"),
        b_shared_seq_len=torch.full((5,), 7, dtype=torch.int32, device="cuda"),
        b_shared_radix_node_id=torch.full((5,), 10, dtype=torch.int64, device="cuda"),
        mem_indexes=torch.arange(5, dtype=torch.int32, device="cuda"),
        mem_indexes_cpu=torch.arange(5, dtype=torch.int32, device="cpu"),
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []} for _ in range(5)],
    )
    selected_row_mask = torch.ones((5,), dtype=torch.int32, device="cuda")

    compacted_input = dynamic_mtp_utils._compact_decode_model_input(
        model_input=model_input,
        selected_row_mask=selected_row_mask,
        dynamic_batch_size=5,
    )
    torch.cuda.synchronize()

    assert torch.equal(compacted_input.b_req_idx.cpu(), torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32))
    assert torch.equal(compacted_input.b_mtp_index.cpu(), torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32))
    assert torch.equal(compacted_input.b_shared_seq_len.cpu(), torch.full((5,), 7, dtype=torch.int32))
    assert torch.equal(compacted_input.b_shared_radix_node_id.cpu(), torch.full((5,), 10, dtype=torch.int64))


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
        select = sample_dynamic_mtp_row_mask(
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
    select = sample_dynamic_mtp_row_mask(
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
        select = sample_dynamic_mtp_row_mask(
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
    select = sample_dynamic_mtp_row_mask(
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
    select = sample_dynamic_mtp_row_mask(
        dynamic_batch_size=2,
        b_req_idx=b_req_idx,
        req_to_next_token_scores=scores.clone(),
        max_draft_step=max_draft_step,
    )
    _assert_topk_mask(select, flat_scores, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

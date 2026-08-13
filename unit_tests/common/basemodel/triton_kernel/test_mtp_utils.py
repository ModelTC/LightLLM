import json
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from lightllm.common.basemodel.batch_objs import ModelInput
from lightllm.common.basemodel.triton_kernel import mtp_utils
from lightllm.utils.envs_utils import get_env_start_args


def test_compact_dynamic_spec_model_input(monkeypatch):
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
    mtp_utils.get_diverse_max_batch_shared_group_size.cache_clear()

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
        b_mark_shared_group=torch.tensor([0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4], dtype=torch.int32, device="cuda"),
        mem_indexes=torch.arange(12, dtype=torch.int32, device="cuda") + 100,
        mem_indexes_cpu=torch.arange(12, dtype=torch.int32, device="cpu") + 100,
        is_prefill=False,
        draft_step=3,
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

    compacted_input, selected_row_mask = mtp_utils.prepare_dynamic_spec_model_input(
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
        compacted_input.b_mark_shared_group.cpu(), torch.tensor([0, 0, 3, 1, 0, 0, 0, 4], dtype=torch.int32)
    )
    assert torch.equal(
        compacted_input.mem_indexes.cpu(), torch.tensor([100, 101, 102, 104, 108, 109, 110, 111], dtype=torch.int32)
    )
    # The hot path intentionally keeps the CPU copy unfiltered to avoid a GPU-to-CPU
    # synchronization. The router frees rejected indexes after its async mask copy.
    assert torch.equal(compacted_input.mem_indexes_cpu, torch.arange(12, dtype=torch.int32) + 100)

    expected_hiddens = (torch.arange(12 * 5, dtype=torch.float32).reshape(12, 5) + 0.5)[expected_selected_rows]
    assert torch.equal(compacted_input.mtp_draft_input_hiddens.cpu(), expected_hiddens)


def test_compaction_rebuilds_b_mark_shared_group_by_max_batch_shared_group_size(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_MAX_BATCH_SHARED_GROUP_SIZE", "3")
    mtp_utils.get_diverse_max_batch_shared_group_size.cache_clear()

    model_input = ModelInput(
        batch_size=5,
        total_token_num=36,
        max_q_seq_len=1,
        max_kv_seq_len=6,
        input_ids=None,
        b_req_idx=torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"),
        b_mtp_index=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([3, 4, 5, 6, 7], dtype=torch.int32, device="cuda"),
        b_shared_seq_len=None,
        b_mark_shared_group=torch.tensor([0, 0, 0, 0, 5], dtype=torch.int32, device="cuda"),
        mem_indexes=torch.arange(5, dtype=torch.int32, device="cuda"),
        mem_indexes_cpu=torch.arange(5, dtype=torch.int32, device="cpu"),
        draft_step=4,
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []} for _ in range(5)],
    )
    selected_row_mask = torch.ones((5,), dtype=torch.int32, device="cuda")

    compacted_input = mtp_utils._compact_decode_model_input(
        model_input=model_input,
        selected_row_mask=selected_row_mask,
        dynamic_batch_size=5,
    )
    torch.cuda.synchronize()

    assert torch.equal(compacted_input.b_req_idx.cpu(), torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32))
    assert torch.equal(compacted_input.b_mtp_index.cpu(), torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32))
    assert torch.equal(compacted_input.b_mark_shared_group.cpu(), torch.tensor([0, 0, 3, 0, 2], dtype=torch.int32))


def test_mtp_verify_scatter_and_start_locations():
    req_to_next_token_ids = torch.tensor(
        [[1, 2, -2, -1, -1], [1, 2, 0, -1, -1], [1, 3, 4, 4, 5]],
        dtype=torch.int64,
        device="cuda",
    )
    b_req_idx = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int32, device="cuda")
    b_mtp_index = torch.tensor([0, 1, 0, 1, 2], dtype=torch.int32, device="cuda")
    b_req_mtp_start_loc = mtp_utils.gen_b_req_mtp_start_loc(b_mtp_index, num_reqs=2)
    new_next_token_ids = torch.tensor([1, 4, 2, 4, 13], dtype=torch.int64, device="cuda")
    all_next_token_ids = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        dtype=torch.int64,
        device="cuda",
    )
    req_to_next_token_scores = torch.full((3, 5), -1.0, dtype=torch.float32, device="cuda")
    schedule_scores = torch.tensor(
        [[0.9, 0.8], [0.8, 0.7], [0.7, 0.6], [0.6, 0.5], [0.5, 0.4]],
        dtype=torch.float32,
        device="cuda",
    )

    spec_accept_len, accepted_index = mtp_utils.mtp_verify(
        req_to_next_token_ids, b_req_mtp_start_loc, new_next_token_ids, b_req_idx
    )
    mtp_utils.mtp_scatter_next_token_ids(
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        all_next_token_ids=all_next_token_ids,
        b_req_idx=b_req_idx,
        spec_accept_len=spec_accept_len,
        req_to_next_token_scores=req_to_next_token_scores,
        schedule_scores=schedule_scores,
    )
    torch.cuda.synchronize()

    assert torch.equal(b_req_mtp_start_loc.cpu(), torch.tensor([0, 2], dtype=torch.int64))
    assert torch.equal(spec_accept_len.cpu(), torch.tensor([1, 1], dtype=torch.int32))
    assert torch.equal(accepted_index.cpu(), torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32))
    assert torch.equal(
        req_to_next_token_ids.cpu(),
        torch.tensor(
            [[1, 2, 3, 1, 1], [1, 2, 0, -1, -1], [7, 8, 9, 1, 1]],
            dtype=torch.int64,
        ),
    )
    torch.testing.assert_close(
        req_to_next_token_scores.cpu(),
        torch.tensor(
            [[1.0, 0.9, 0.8, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0, -1.0], [1.0, 0.7, 0.6, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )


def test_mtp_scatter_handles_zero_draft_step():
    req_to_next_token_ids = torch.full((1, 4), -1, dtype=torch.int64, device="cuda")
    req_to_next_token_scores = torch.full((1, 4), -1.0, dtype=torch.float32, device="cuda")

    mtp_utils.mtp_scatter_next_token_ids(
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_mtp_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
        all_next_token_ids=torch.tensor([[42]], dtype=torch.int64, device="cuda"),
        b_req_idx=torch.tensor([0], dtype=torch.int32, device="cuda"),
        spec_accept_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
        req_to_next_token_scores=req_to_next_token_scores,
        schedule_scores=torch.empty((1, 0), dtype=torch.float32, device="cuda"),
    )
    torch.cuda.synchronize()

    assert req_to_next_token_ids.cpu().tolist() == [[42, 1, 1, 1]]
    assert req_to_next_token_scores.cpu().tolist() == [[1.0, 0.0, 0.0, 0.0]]

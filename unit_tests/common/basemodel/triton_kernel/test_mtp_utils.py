import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from lightllm.common.basemodel.triton_kernel import mtp_utils


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

    mtp_accept_len, accepted_index = mtp_utils.mtp_verify(
        req_to_next_token_ids, b_req_mtp_start_loc, new_next_token_ids, b_req_idx
    )
    mtp_utils.mtp_scatter_next_token_ids(
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        all_next_token_ids=all_next_token_ids,
        b_req_idx=b_req_idx,
        mtp_accept_len=mtp_accept_len,
        req_to_next_token_scores=req_to_next_token_scores,
        schedule_scores=schedule_scores,
    )
    torch.cuda.synchronize()

    assert torch.equal(b_req_mtp_start_loc.cpu(), torch.tensor([0, 2], dtype=torch.int64))
    assert torch.equal(mtp_accept_len.cpu(), torch.tensor([1, 1], dtype=torch.int32))
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
        mtp_accept_len=torch.tensor([1], dtype=torch.int32, device="cuda"),
        req_to_next_token_scores=req_to_next_token_scores,
        schedule_scores=torch.empty((1, 0), dtype=torch.float32, device="cuda"),
    )
    torch.cuda.synchronize()

    assert req_to_next_token_ids.cpu().tolist() == [[42, 1, 1, 1]]
    assert req_to_next_token_scores.cpu().tolist() == [[1.0, 0.0, 0.0, 0.0]]

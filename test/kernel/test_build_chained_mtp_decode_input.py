import pytest
import torch

from lightllm.common.basemodel.triton_kernel.build_chained_mtp_decode_input import (
    build_chained_mtp_decode_input_inplace,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_chained_mtp_decode_input_inplace_shifts_accepted_tokens_and_keeps_tail():
    input_ids = torch.tensor(
        [
            10,
            11,
            12,
            20,
            30,
            31,
            40,
            41,
            42,
            43,
            90,
            91,
        ],
        dtype=torch.int64,
        device="cuda",
    )
    draft_token_ids = torch.tensor(
        [
            100,
            101,
            102,
            200,
            300,
            301,
            400,
            401,
            402,
            403,
            900,
            901,
        ],
        dtype=torch.int64,
        device="cuda",
    )
    b_req_mtp_start_loc = torch.tensor([0, 3, 4, 6], dtype=torch.int32, device="cuda")
    accept_len = torch.tensor([3, 1, 2, 4], dtype=torch.int32, device="cuda")

    result = build_chained_mtp_decode_input_inplace(
        input_ids=input_ids,
        draft_token_ids=draft_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        accept_len=accept_len,
    )

    expected = torch.tensor(
        [
            11,
            12,
            102,
            200,
            31,
            301,
            41,
            42,
            43,
            403,
            900,
            901,
        ],
        dtype=torch.int64,
        device="cuda",
    )
    assert result is draft_token_ids
    torch.testing.assert_close(draft_token_ids, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_chained_mtp_decode_input_inplace_handles_empty_request_set():
    input_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
    draft_token_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
    b_req_mtp_start_loc = torch.empty((0,), dtype=torch.int32, device="cuda")
    accept_len = torch.empty((0,), dtype=torch.int32, device="cuda")

    result = build_chained_mtp_decode_input_inplace(
        input_ids=input_ids,
        draft_token_ids=draft_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        accept_len=accept_len,
    )

    assert result is draft_token_ids
    assert draft_token_ids.numel() == 0

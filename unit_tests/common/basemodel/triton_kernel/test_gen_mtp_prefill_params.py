import torch
import pytest
import numpy as np
from lightllm.utils.log_utils import init_logger
from lightllm.common.basemodel.triton_kernel.gen_mtp_prefill_params import gen_mtp_new_input_ids


def test_gen_mtp_new_input_ids_empty_batch():
    input_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
    b_next_token_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
    b_seq_len = torch.empty((0,), dtype=torch.int32, device="cuda")

    new_input_ids = gen_mtp_new_input_ids(
        input_ids=input_ids,
        b_next_token_ids=b_next_token_ids,
        b_seq_len=b_seq_len,
    )

    assert new_input_ids.is_cuda
    assert new_input_ids.dtype == torch.int64
    assert new_input_ids.shape == (0,)


def test_gen_mtp_new_input_ids_0():
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).long().cuda()
    b_next_token_ids = torch.tensor([10, 11, 12]).long().cuda()
    b_seq_len = torch.tensor([3, 3, 3]).int().cuda()
    expected_output = torch.tensor([2, 3, 10, 5, 6, 11, 8, 9, 12]).long().cuda()
    new_input_ids = gen_mtp_new_input_ids(input_ids, b_next_token_ids, b_seq_len)
    assert torch.equal(new_input_ids, expected_output)


def test_gen_mtp_new_input_ids_1():
    input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).long().cuda()
    b_next_token_ids = torch.tensor([10, 11, 12]).long().cuda()
    b_seq_len = torch.tensor([3, 3, 3]).int().cuda()
    b_ready_cache_len = torch.tensor([1, 1, 1]).int().cuda()
    expected_output = torch.tensor([2, 10, 4, 11, 6, 12]).long().cuda()
    new_input_ids = gen_mtp_new_input_ids(input_ids, b_next_token_ids, b_seq_len, b_ready_cache_len=b_ready_cache_len)
    assert torch.equal(new_input_ids, expected_output)


if __name__ == "__main__":
    pytest.main()

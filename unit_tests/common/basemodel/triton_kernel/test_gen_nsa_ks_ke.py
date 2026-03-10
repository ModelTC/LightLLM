import torch
import pytest
from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke, gen_same_req_mark


def test_gen_nsa_ks_ke_basic():
    """Test basic functionality of gen_nsa_ks_ke with simple inputs."""
    # Setup test data
    b_seq_len = torch.tensor(
        [
            10,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    b_q_seq_len = torch.tensor(
        [
            5,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    b_req_idx = torch.tensor(
        [
            1,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    q_token_num = b_q_seq_len.sum().item()

    req_to_token_index = torch.arange(0, 1000).cuda().view(100, -1)
    ragged_mem_index = torch.empty_like(req_to_token_index.view(-1))

    ks, ke, lengths = gen_nsa_ks_ke(
        b_seq_len, b_q_seq_len, b_req_idx, req_to_token_index, q_token_num, ragged_mem_index
    )

    assert torch.equal(ks, torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"))
    assert torch.equal(ke, torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32, device="cuda"))
    assert torch.equal(lengths, ke - ks)
    assert torch.equal(
        ragged_mem_index[0:10], torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int32, device="cuda")
    )
    return


def test_gen_nsa_ks_ke_batch():
    b_seq_len = torch.tensor([10, 11], dtype=torch.int32, device="cuda")
    b_q_seq_len = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    b_req_idx = torch.tensor([1, 1], dtype=torch.int32, device="cuda")
    q_token_num = b_q_seq_len.sum().item()

    req_to_token_index = torch.arange(0, 1000).cuda().view(10, -1)
    ragged_mem_index = torch.empty_like(req_to_token_index.view(-1))

    ks, ke, lengths = gen_nsa_ks_ke(
        b_seq_len, b_q_seq_len, b_req_idx, req_to_token_index, q_token_num, ragged_mem_index
    )

    assert torch.equal(
        ks,
        torch.tensor(
            [
                0,
                0,
            ],
            dtype=torch.int32,
            device="cuda",
        ),
    )
    assert torch.equal(ke, torch.tensor([10, 11], dtype=torch.int32, device="cuda"))
    assert torch.equal(lengths, ke - ks)
    assert torch.equal(ragged_mem_index[0:11], torch.arange(100, 100 + 11).cuda())


def test_gen_same_req_mark():
    b_req_idx = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int32, device="cuda")
    expected_same_req_mark = torch.tensor([0, 2, 0, 0, 3, 1], dtype=torch.int32, device="cuda")

    same_req_mark = gen_same_req_mark(b_req_idx)

    assert torch.equal(same_req_mark, expected_same_req_mark)


if __name__ == "__main__":
    pytest.main()

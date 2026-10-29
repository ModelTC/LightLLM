import torch
import pytest
from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import (
    gen_nsa_ks_ke,
    gen_same_req_mark,
)


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
        b_seq_len,
        b_q_seq_len,
        b_req_idx,
        req_to_token_index,
        q_token_num,
        ragged_mem_index,
    )

    assert torch.equal(ks, torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"))
    assert torch.equal(ke, torch.tensor([6, 7, 8, 9, 10], dtype=torch.int32, device="cuda"))
    assert torch.equal(lengths, ke - ks)
    assert torch.equal(
        ragged_mem_index[0:10],
        torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=torch.int32, device="cuda"),
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
        b_seq_len,
        b_q_seq_len,
        b_req_idx,
        req_to_token_index,
        q_token_num,
        ragged_mem_index,
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


@pytest.mark.parametrize("context_length", [4096, 32768, 131072, 524288, 1000000])
@pytest.mark.parametrize("verify_width", [1, 4])
def test_parallel_ragged_indices_preserve_mtp_and_hold_rows(context_length, verify_width):
    # Include recycled/nonsequential slots and independent HOLD padding rows.
    hold = 3
    req = torch.tensor(
        [2] * verify_width + [0] * verify_width + [hold] * 4,
        device="cuda",
        dtype=torch.int32,
    )
    seq = torch.tensor(
        list(range(context_length - verify_width + 1, context_length + 1))
        + list(range(4096 - verify_width + 1, 4097))
        + [2] * 4,
        device="cuda",
        dtype=torch.int32,
    )
    q = torch.ones_like(req)
    table = torch.arange(4 * context_length, device="cuda", dtype=torch.int32).reshape(4, context_length)
    capacity = context_length + 4096 + 8
    reference_ragged = torch.full((capacity,), -1, device="cuda", dtype=torch.int32)
    parallel_ragged = reference_ragged.clone()
    reference = gen_nsa_ks_ke(seq, q, req, table, len(req), reference_ragged, hold_req_idx=hold)
    actual = gen_nsa_ks_ke(
        seq,
        q,
        req,
        table,
        len(req),
        parallel_ragged,
        hold_req_idx=hold,
        max_seq_len=context_length,
    )
    for observed, expected in zip(actual, reference):
        assert torch.equal(observed, expected)
    assert torch.equal(parallel_ragged, reference_ragged)
    assert torch.equal(parallel_ragged[:context_length], table[2, :context_length])
    assert torch.equal(parallel_ragged[context_length : context_length + 4096], table[0, :4096])


def test_parallel_ragged_metadata_matches_prefill_and_recycled_layout():
    table = torch.arange(4 * 131072, device="cuda", dtype=torch.int32).reshape(4, -1)
    q = torch.tensor([8192, 256], device="cuda", dtype=torch.int32)
    req = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
    for lengths in ([131072, 32768], [32768, 65536], [65536, 131072]):
        seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        ragged = torch.empty(sum(lengths), device="cuda", dtype=torch.int32)
        ks, ke, actual_lengths = gen_nsa_ks_ke(seq, q, req, table, 8192 + 256, ragged, max_seq_len=max(lengths))
        expected_lengths = torch.cat(
            (
                torch.arange(
                    lengths[0] - 8192 + 1,
                    lengths[0] + 1,
                    device="cuda",
                    dtype=torch.int32,
                ),
                torch.arange(
                    lengths[1] - 256 + 1,
                    lengths[1] + 1,
                    device="cuda",
                    dtype=torch.int32,
                ),
            )
        )
        expected_ks = torch.cat(
            (
                torch.zeros(8192, device="cuda", dtype=torch.int32),
                torch.full((256,), lengths[0], device="cuda", dtype=torch.int32),
            )
        )
        assert torch.equal(actual_lengths, expected_lengths)
        assert torch.equal(ks, expected_ks)
        assert torch.equal(ke, expected_ks + expected_lengths)
        assert torch.equal(ragged, torch.cat((table[2, : lengths[0]], table[0, : lengths[1]])))


if __name__ == "__main__":
    pytest.main()

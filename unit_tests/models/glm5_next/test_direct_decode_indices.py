"""Decode can read K-pool slots from req_to_token instead of a per-forward ragged copy."""

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools, expand_topk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HOLD = 5


def _mtp_decode_batch(context_lengths, verify_width, hold_rows, table_width):
    """Rows of a fixed-width MTP verify batch plus HOLD padding rows."""
    req_ids = [1, 4, 2][: len(context_lengths)]
    req, seq = [], []
    for r, length in zip(req_ids, context_lengths):
        req += [r] * verify_width
        seq += list(range(length - verify_width + 1, length + 1))
    req += [HOLD] * hold_rows
    seq += [1] * hold_rows
    # Recycled, nonsequential slots: every request owns a random permutation.
    table = torch.randperm(6 * table_width, device="cuda", dtype=torch.int32).view(6, table_width)
    tensor = lambda x: torch.tensor(x, device="cuda", dtype=torch.int32)
    return tensor(req), tensor(seq), table


@pytest.mark.parametrize("context_lengths", [[9, 7, 5], [4099, 131, 4], [40000, 70001]])
@pytest.mark.parametrize("verify_width", [1, 4])
@pytest.mark.parametrize("hold_rows", [0, 3])
def test_lengths_match_without_ragged_copy(context_lengths, verify_width, hold_rows):
    req, seq, table = _mtp_decode_batch(context_lengths, verify_width, hold_rows, max(context_lengths) + 8)
    q = torch.ones_like(req)
    ragged = torch.full((int(seq.sum()) + 8,), -1, device="cuda", dtype=torch.int32)
    reference = gen_nsa_ks_ke(seq, q, req, table, len(req), ragged, hold_req_idx=HOLD, max_seq_len=int(seq.max()))
    direct = gen_nsa_ks_ke(seq, q, req, table, len(req), None, hold_req_idx=HOLD, copy_ragged=False)
    for observed, expected in zip(direct, reference):
        assert torch.equal(observed, expected)


@pytest.mark.parametrize("context_lengths", [[9, 7, 5], [4099, 131, 4], [40000, 70001]])
@pytest.mark.parametrize("verify_width", [1, 4])
@pytest.mark.parametrize("hold_rows", [0, 3])
@pytest.mark.parametrize("dense", [False, True])
def test_expand_topk_direct_matches_ragged(context_lengths, verify_width, hold_rows, dense):
    torch.manual_seed(7)
    topk = 2048
    req, seq, table = _mtp_decode_batch(context_lengths, verify_width, hold_rows, max(context_lengths) + 8)
    q = torch.ones_like(req)
    ragged = torch.full((int(seq.sum()) + 8,), -1, device="cuda", dtype=torch.int32)
    ks, _, lengths = gen_nsa_ks_ke(seq, q, req, table, len(req), ragged, hold_req_idx=HOLD)
    pools = (lengths // 4).clamp_min(1)
    groups = (torch.rand(len(req), topk // 4, device="cuda") * pools[:, None]).to(torch.int32)
    groups[:, -3:] = -1
    groups = None if dense else groups
    expected = expand_topk(groups, lengths, ks, ragged, topk, dense=dense)
    actual = expand_topk(groups, lengths, ks, None, topk, dense=dense, req_idx=req, req_to_token_indexs=table)
    for observed, reference in zip(actual, expected):
        assert torch.equal(observed, reference)


@pytest.mark.parametrize("verify_width", [1, 4])
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_compress_pools_direct_matches_ragged(verify_width, cuda_graph):
    torch.manual_seed(11)
    # Every pool phase appears across rows so some rows close a pool this step.
    req, seq, table = _mtp_decode_batch([13, 8, 6], verify_width, 2, 32)
    rows = len(req)
    q = torch.ones_like(req)
    ragged = torch.full((int(seq.sum()) + 8,), -1, device="cuda", dtype=torch.int32)
    ks, _, lengths = gen_nsa_ks_ke(seq, q, req, table, rows, ragged, hold_req_idx=HOLD)
    raw = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(4, 128, device="cuda")
    tail0 = torch.randn(HOLD + 1, 4, 256, device="cuda", dtype=torch.bfloat16)
    packed0 = torch.zeros(6 * 32, 1, 132, device="cuda", dtype=torch.uint8)
    cu_q_lens = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    mtp_index = torch.tensor([i % verify_width for i in range(rows - 2)] + [0, 0], device="cuda", dtype=torch.int32)

    def run(direct):
        tail, packed = tail0.clone(), packed0.clone()

        def call():
            compress_pools(
                raw,
                tail,
                packed,
                ape,
                lengths,
                ks,
                None if direct else ragged,
                req,
                cu_q_lens,
                seq,
                max_q_len=1,
                mtp_index=mtp_index,
                req_to_token_indexs=table if direct else None,
            )

        if cuda_graph:
            saved_tail, saved_packed = tail.clone(), packed.clone()
            call()
            torch.cuda.synchronize()
            tail.copy_(saved_tail)
            packed.copy_(saved_packed)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            graph.replay()
        else:
            call()
        torch.cuda.synchronize()
        return tail, packed

    expected_tail, expected_packed = run(direct=False)
    actual_tail, actual_packed = run(direct=True)
    assert torch.equal(actual_tail, expected_tail)
    assert torch.equal(actual_packed, expected_packed)
    # At least one pool was written; otherwise the comparison is vacuous.
    assert not torch.equal(expected_packed, packed0)

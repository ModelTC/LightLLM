import pytest
import torch

from lightllm.common.basemodel.triton_kernel.mtp_asd import mtp_asd_verify
from lightllm.common.basemodel.triton_kernel.mtp_utils import mtp_verify

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ASD verify kernel requires CUDA")


def _build_case(device="cuda"):
    """One request with 2 draft rows + 1 bonus row; known per-row regrets.

    Layout: committed token 7, draft tokens [3, 5].
    row0: argmax 9 (logit 10.0), draft 3 (logit 8.0) -> regret 2.0
    row1: argmax 9 (logit 10.0), draft 5 (logit 7.4) -> regret 2.6
    row2: bonus row, argmax 6
    """
    req_to_next_token_ids = torch.tensor([[7, 3, 5, -1, -1]], dtype=torch.int64, device=device)
    b_req_idx = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
    b_mtp_index = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    b_req_mtp_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
    vocab = 10
    logits = torch.full((3, vocab), -10.0, dtype=torch.float32, device=device)
    logits[0, 9] = 10.0
    logits[0, 3] = 8.0
    logits[1, 9] = 10.0
    logits[1, 5] = 7.4
    logits[2, 6] = 10.0
    new_next_token_ids = torch.tensor([9, 9, 6], dtype=torch.int64, device=device)
    req_to_asd_cum_regret = torch.zeros(1, dtype=torch.float32, device=device)
    return (
        req_to_next_token_ids,
        b_req_mtp_start_loc,
        new_next_token_ids,
        b_req_idx,
        b_mtp_index,
        logits,
        req_to_asd_cum_regret,
    )


def test_asd_zero_budget_equals_strict():
    """Gold standard: B=0 (and m=0) must reproduce strict verification exactly."""
    (
        req_to_next_token_ids,
        b_req_mtp_start_loc,
        new_next_token_ids,
        b_req_idx,
        b_mtp_index,
        logits,
        req_to_asd_cum_regret,
    ) = _build_case()

    strict_len, strict_index = mtp_verify(
        req_to_next_token_ids, b_req_mtp_start_loc, new_next_token_ids.clone(), b_req_idx
    )
    for budget, max_mismatch in [(0.0, 2), (100.0, 0)]:
        asd_ids = new_next_token_ids.clone()
        asd_len, asd_index = mtp_asd_verify(
            req_to_next_token_ids=req_to_next_token_ids,
            b_req_mtp_start_loc=b_req_mtp_start_loc,
            new_next_token_ids=asd_ids,
            b_req_idx=b_req_idx,
            b_mtp_index=b_mtp_index,
            logits=logits,
            req_to_asd_cum_regret=req_to_asd_cum_regret,
            asd_budget=budget,
            asd_local_ratio=100.0,
            asd_max_mismatch=max_mismatch,
        )
        assert torch.equal(asd_len, strict_len)
        assert torch.equal(asd_index, strict_index)
        assert torch.equal(asd_ids, new_next_token_ids)  # no relaxed commit under strict equivalence
        assert req_to_asd_cum_regret.item() == 0.0


def test_asd_relaxed_acceptance_hand_computed():
    """row0 (regret 2.0) fits the budget and is relaxed-accepted with the draft token;
    row1 (cumulative 4.6) exceeds B=3.0 and stops acceptance."""
    (
        req_to_next_token_ids,
        b_req_mtp_start_loc,
        new_next_token_ids,
        b_req_idx,
        b_mtp_index,
        logits,
        req_to_asd_cum_regret,
    ) = _build_case()

    asd_ids = new_next_token_ids.clone()
    accept_len, accepted_index = mtp_asd_verify(
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=asd_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        logits=logits,
        req_to_asd_cum_regret=req_to_asd_cum_regret,
        asd_budget=3.0,
        asd_local_ratio=100.0,
        asd_max_mismatch=2,
    )
    assert accept_len.tolist() == [2]
    assert accepted_index.tolist() == [1, 1, 0]
    assert asd_ids.tolist() == [3, 9, 6]  # relaxed row commits the draft token 3
    assert req_to_asd_cum_regret.item() == pytest.approx(2.0)


def test_asd_local_ratio_gate():
    """The local gate rejects a late position even when the budget is ample.

    row1 regret 2.6 with suffix value K - i = 2 - 1 = 1: 2.6 / 1 > g=1.5 -> rejected.
    row0 regret 2.0 with suffix 2: 2.0 / 2 = 1.0 <= 1.5 -> accepted.
    """
    (
        req_to_next_token_ids,
        b_req_mtp_start_loc,
        new_next_token_ids,
        b_req_idx,
        b_mtp_index,
        logits,
        req_to_asd_cum_regret,
    ) = _build_case()

    asd_ids = new_next_token_ids.clone()
    accept_len, _ = mtp_asd_verify(
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=asd_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        logits=logits,
        req_to_asd_cum_regret=req_to_asd_cum_regret,
        asd_budget=100.0,
        asd_local_ratio=1.5,
        asd_max_mismatch=5,
    )
    assert accept_len.tolist() == [2]
    assert asd_ids.tolist() == [3, 9, 6]


def test_asd_max_mismatch_cap():
    """m=1 lets only the first relaxed token through even with ample budget."""
    (
        req_to_next_token_ids,
        b_req_mtp_start_loc,
        new_next_token_ids,
        b_req_idx,
        b_mtp_index,
        logits,
        req_to_asd_cum_regret,
    ) = _build_case()

    asd_ids = new_next_token_ids.clone()
    accept_len, _ = mtp_asd_verify(
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=asd_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        logits=logits,
        req_to_asd_cum_regret=req_to_asd_cum_regret,
        asd_budget=100.0,
        asd_local_ratio=100.0,
        asd_max_mismatch=1,
    )
    assert accept_len.tolist() == [2]
    assert asd_ids.tolist() == [3, 9, 6]


def test_asd_all_accepted_commits_bonus():
    """All draft rows feasible: accept_len == req_mtp_num, bonus row keeps the target id."""
    (
        req_to_next_token_ids,
        b_req_mtp_start_loc,
        new_next_token_ids,
        b_req_idx,
        b_mtp_index,
        logits,
        req_to_asd_cum_regret,
    ) = _build_case()

    asd_ids = new_next_token_ids.clone()
    accept_len, accepted_index = mtp_asd_verify(
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=asd_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        logits=logits,
        req_to_asd_cum_regret=req_to_asd_cum_regret,
        asd_budget=100.0,
        asd_local_ratio=100.0,
        asd_max_mismatch=2,
    )
    assert accept_len.tolist() == [3]
    assert accepted_index.tolist() == [1, 1, 1]
    assert asd_ids.tolist() == [3, 5, 6]  # both draft tokens committed, bonus row keeps argmax 6
    assert req_to_asd_cum_regret.item() == pytest.approx(4.6)


def test_asd_budget_persists_across_steps():
    """The ledger carries over: regret spent in step 1 shrinks the allowance of step 2."""
    (
        req_to_next_token_ids,
        b_req_mtp_start_loc,
        new_next_token_ids,
        b_req_idx,
        b_mtp_index,
        logits,
        req_to_asd_cum_regret,
    ) = _build_case()

    mtp_asd_verify(
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=new_next_token_ids.clone(),
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        logits=logits,
        req_to_asd_cum_regret=req_to_asd_cum_regret,
        asd_budget=3.0,
        asd_local_ratio=100.0,
        asd_max_mismatch=2,
    )
    assert req_to_asd_cum_regret.item() == pytest.approx(2.0)

    # Step 2: budget_used=2.0, row0 needs 2.0 more -> 4.0 > B=3.0 -> rejected at position 0.
    asd_ids = new_next_token_ids.clone()
    accept_len, accepted_index = mtp_asd_verify(
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        new_next_token_ids=asd_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        logits=logits,
        req_to_asd_cum_regret=req_to_asd_cum_regret,
        asd_budget=3.0,
        asd_local_ratio=100.0,
        asd_max_mismatch=2,
    )
    assert accept_len.tolist() == [1]
    assert accepted_index.tolist() == [1, 0, 0]
    assert asd_ids.tolist() == [9, 9, 6]  # nothing relaxed
    assert req_to_asd_cum_regret.item() == pytest.approx(2.0)  # unchanged

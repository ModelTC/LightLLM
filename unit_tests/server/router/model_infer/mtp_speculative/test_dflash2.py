from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mtp_speculative.dflash2 import (
    _rejection_sample_from_probs,
    save_dflash2_proposal_state,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import (
    DFlash2SpecProposal,
    SpecProposal,
)


@pytest.mark.parametrize(
    "first,second,has_proposal,expected",
    [
        ([1.0, 0, 0], [0, 1.0, 0], False, [0]),
        ([1.0, 0, 0], [0, 1.0, 0], True, [0, 1, 2]),
        ([0.5, 0.5, 0], [0, 1.0, 0], True, [1]),
        ([1.0, 0, 0], [0, 0.5, 0.5], True, [0, 2]),
    ],
    ids=["first_round", "all_accepted", "first_rejected", "middle_rejected"],
)
def test_rejection_prefix_and_logprobs(first, second, has_proposal, expected):
    probs = torch.tensor([[first, second, [0, 0, 1.0]]])
    raw = torch.tensor([[[0.2, 0.3, 0.5], [0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]])
    q = torch.tensor([[[1.0, 0, 0], [0, 1.0, 0]]]) * has_proposal
    tokens, logprobs, lengths, mask = _rejection_sample_from_probs(
        sampling_probs=probs,
        raw_probs=raw,
        proposal_tokens=torch.tensor([[0, 1]]),
        candidate_ids=torch.arange(3).expand(1, 2, 3),
        q_rows=q,
        request_reqs=[SimpleNamespace(generator=None)],
        acceptance_uniforms=torch.tensor([[0.9, 0.9]]),
    )
    assert lengths.tolist() == [len(expected)]
    assert mask.tolist() == [int(i < len(expected)) for i in range(3)]
    assert tokens[mask.bool()].tolist() == expected
    expected_logprobs = raw[0, torch.arange(len(expected)), torch.tensor(expected)].log()
    torch.testing.assert_close(logprobs[mask.bool()], expected_logprobs)


def test_rejection_recovers_target_distribution():
    # A non-delta q checks both p/q acceptance and residual sampling together.
    count = 20000
    p = torch.tensor([0.6, 0.3, 0.1])
    q = torch.tensor([0.1, 0.3, 0.6])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        tokens, _, _, _ = _rejection_sample_from_probs(
            sampling_probs=p.expand(count, 2, 3),
            raw_probs=p.expand(count, 2, 3),
            proposal_tokens=torch.multinomial(q, count, replacement=True).view(count, 1),
            candidate_ids=torch.arange(3).expand(count, 1, 3),
            q_rows=q.expand(count, 1, 3),
            request_reqs=[SimpleNamespace(generator=None)] * count,
        )
    frequencies = torch.bincount(tokens.view(count, 2)[:, 0], minlength=3).float() / count
    torch.testing.assert_close(frequencies, p, atol=0.02, rtol=0)


@pytest.mark.parametrize("invalid", [None, "type", "width"])
def test_proposal_state_preserves_request_slots(invalid):
    manager = SimpleNamespace(
        mtp_mode="dflash2",
        req_to_dflash2_candidate_ids=torch.zeros(5, 2, 16, dtype=torch.int64),
        req_to_dflash2_q_probs=torch.zeros(5, 2, 16),
    )
    backend = SimpleNamespace(model=SimpleNamespace(req_manager=SimpleNamespace(req_sampling_params_manager=manager)))
    candidates = torch.arange(64).reshape(2, 2, 16)
    q = torch.zeros(2, 2, 16)
    q[0, :, 0], q[1, :, 1] = 1, 1
    proposal = DFlash2SpecProposal(token_ids=candidates[:, :, 0], candidate_ids=candidates, q_probs=q)
    req_ids, starts = torch.tensor([3, 3, 3, 1, 1, 1]), torch.tensor([0, 3])
    if invalid is not None:
        if invalid == "type":
            proposal = SpecProposal(token_ids=proposal.token_ids)
        else:
            proposal.candidate_ids, proposal.q_probs = candidates[:, :1], q[:, :1]
        with pytest.raises(TypeError if invalid == "type" else AssertionError, match="DFlash2 requires"):
            save_dflash2_proposal_state(backend, proposal, req_ids, starts)
        assert not manager.req_to_dflash2_candidate_ids.any()
        assert not manager.req_to_dflash2_q_probs.any()
        return
    save_dflash2_proposal_state(backend, proposal, req_ids, starts)
    torch.testing.assert_close(manager.req_to_dflash2_candidate_ids[[3, 1]], candidates)
    torch.testing.assert_close(manager.req_to_dflash2_q_probs[[3, 1]], q)
    assert not manager.req_to_dflash2_candidate_ids[[0, 2, 4]].any()
    assert not manager.req_to_dflash2_q_probs[[0, 2, 4]].any()

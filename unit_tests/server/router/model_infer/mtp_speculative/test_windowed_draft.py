from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash import DFlashProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.dspark import DSparkProposer
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


@pytest.mark.parametrize("proposer_class", [DFlashProposer, DSparkProposer])
def test_windowed_prefill_commits_features_without_token_cache_forward(proposer_class):
    committed = []
    draft = SimpleNamespace(uses_windowed_draft_kv=True, commit_features=lambda *args: committed.append(args))
    proposer = proposer_class(backend=SimpleNamespace(draft_models=[draft]), enable_dynmaic_mtp=False)
    model_input = SimpleNamespace(
        is_prefill=True,
        b_position_delta=None,
        b_req_idx=torch.tensor([3, 5], dtype=torch.int32),
        input_ids=torch.arange(5),
    )
    hidden = torch.randn(5, 8)
    proposer.fill_draft_model_kv_state(
        target_model_input=model_input,
        target_model_output=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=hidden)),
        target_next_token_ids=torch.tensor([11, 13]),
    )
    assert len(committed) == 1
    assert committed[0][0] is model_input
    assert committed[0][1] is hidden


@pytest.mark.parametrize("proposer_class", [DFlashProposer, DSparkProposer])
def test_windowed_proposal_commits_accepted_rows_and_uses_page_managed_input(monkeypatch, proposer_class):
    committed = []
    forwarded = []
    token_ids = torch.tensor([30, 31, 32, 40, 41, 42])
    block_output = SimpleNamespace(mtp_collector=SimpleNamespace(draft_token_ids=token_ids))

    def forward(model_input):
        forwarded.append(model_input)
        return block_output

    draft = SimpleNamespace(
        uses_windowed_draft_kv=True,
        block_size=3,
        mask_token_id=99,
        commit_features=lambda *args: committed.append(args),
        forward=forward,
    )
    backend = SimpleNamespace(draft_models=[draft], _gen_argmax_token_ids=lambda _: token_ids)
    proposer = proposer_class(backend=backend, enable_dynmaic_mtp=False)
    monkeypatch.setattr(
        g_pin_mem_manager,
        "get_const_gpu_tensor",
        lambda *, shape, fill_value, dtype, **_: torch.full(shape, fill_value, dtype=dtype),
    )
    hidden = torch.randn(5, 8)
    starts = torch.tensor([0, 3], dtype=torch.int32)
    accepted = torch.tensor([2, 1], dtype=torch.int32)
    model_input = SimpleNamespace(
        is_prefill=False,
        batch_size=5,
        max_kv_seq_len=9,
        input_ids=torch.tensor([10, 11, 12, 20, 21]),
        b_req_idx=torch.tensor([7, 7, 7, 9, 9], dtype=torch.int32),
        b_mtp_index=torch.arange(5, dtype=torch.int32),
        b_seq_len=torch.tensor([4, 5, 6, 8, 9], dtype=torch.int32),
        b_position_delta=torch.tensor([1, 1, 1, 2, 2], dtype=torch.int32),
        b_shared_seq_len=torch.tensor([3, 3, 3, 6, 6], dtype=torch.int32),
        b_shared_radix_node_id=torch.tensor([17, 17, 17, 19, 19]),
        mtp_draft_input_hiddens=None,
    )

    proposal = proposer.propose_next(
        target_model_input=model_input,
        target_model_output=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=hidden)),
        target_next_token_ids=model_input.input_ids,
        b_req_mtp_start_loc=starts,
        draft_step=2,
        accept_len=accepted,
    )

    assert len(committed) == len(forwarded) == 1
    assert committed[0][0] is model_input
    assert committed[0][1] is hidden
    assert committed[0][2] is starts
    assert committed[0][3] is accepted
    block_input = forwarded[0]
    assert block_input is not model_input
    assert block_input.mtp_draft_input_hiddens is None
    assert model_input.mtp_draft_input_hiddens is None
    assert torch.equal(block_input.input_ids, torch.tensor([11, 99, 99, 20, 99, 99]))
    assert torch.equal(block_input.b_seq_len, torch.tensor([6, 7, 8, 9, 10, 11], dtype=torch.int32))
    assert torch.equal(proposal.token_ids, torch.tensor([[30, 31], [40, 41]]))

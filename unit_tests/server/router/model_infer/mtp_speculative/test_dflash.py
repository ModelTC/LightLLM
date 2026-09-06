from types import SimpleNamespace

import torch

from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash import DFlashProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import DFlashSpecProposal
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


def test_dflash_prefill_uses_a_shallow_copy_for_target_hidden():
    forwarded_inputs = []
    draft_model = SimpleNamespace(forward=forwarded_inputs.append)
    proposer = DFlashProposer(
        backend=SimpleNamespace(draft_models=[draft_model]),
        enable_dynmaic_mtp=False,
    )
    model_input = SimpleNamespace(
        is_prefill=True,
        b_position_delta=None,
        b_req_idx=torch.tensor([3, 5], dtype=torch.int32),
        input_ids=torch.arange(5, dtype=torch.int64),
        mtp_draft_input_hiddens=None,
    )
    target_hidden = torch.empty((5, 8))

    proposer.fill_draft_model_kv_state(
        target_model_input=model_input,
        target_model_output=SimpleNamespace(
            mtp_collector=SimpleNamespace(spec_hidden=target_hidden),
        ),
        target_next_token_ids=torch.tensor([11, 13], dtype=torch.int64),
    )

    assert len(forwarded_inputs) == 1
    assert forwarded_inputs[0] is not model_input
    assert forwarded_inputs[0].mtp_draft_input_hiddens is target_hidden
    assert model_input.mtp_draft_input_hiddens is None


def test_dflash_commits_verify_kv_and_builds_parallel_block(monkeypatch):
    block_size = 3
    flat_draft_token_ids = torch.tensor([30, 31, 32, 40, 41, 42], dtype=torch.int64)
    flat_draft_token_probs = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=torch.float16)
    forwarded_inputs = []

    def forward(model_input):
        forwarded_inputs.append(model_input)
        return SimpleNamespace()

    draft_model = SimpleNamespace(
        block_size=block_size,
        mask_token_id=99,
        forward=forward,
    )
    proposer = DFlashProposer(
        backend=SimpleNamespace(
            draft_models=[draft_model],
            _gen_argmax_token_ids_and_prob=lambda _: (flat_draft_token_ids, flat_draft_token_probs),
        ),
        enable_dynmaic_mtp=True,
    )
    extra_mem_indexes_cpu = torch.arange(100, 106, dtype=torch.int32)
    monkeypatch.setattr(mtp_utils, "alloc_mem_indexes", lambda token_num: extra_mem_indexes_cpu)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(
        g_pin_mem_manager,
        "get_const_gpu_tensor",
        lambda *, shape, fill_value, dtype, **_: torch.full(shape, fill_value, dtype=dtype),
    )

    target_hidden = torch.empty((5, 8))
    model_input = SimpleNamespace(
        is_prefill=False,
        batch_size=5,
        total_token_num=40,
        max_q_seq_len=1,
        max_kv_seq_len=9,
        input_ids=torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64),
        b_req_idx=torch.tensor([7, 7, 7, 9, 9], dtype=torch.int32),
        b_mtp_index=torch.arange(5, dtype=torch.int32),
        b_seq_len=torch.tensor([4, 5, 6, 8, 9], dtype=torch.int32),
        b_position_delta=torch.tensor([1, 1, 1, 2, 2], dtype=torch.int32),
        b_shared_seq_len=torch.tensor([3, 3, 3, 6, 6], dtype=torch.int32),
        b_shared_radix_node_id=torch.tensor([17, 17, 17, 19, 19], dtype=torch.int64),
        mem_indexes=torch.arange(5, dtype=torch.int32),
        mem_indexes_cpu=torch.arange(5, dtype=torch.int32),
        multimodal_params=[{"images": [], "audios": []} for _ in range(5)],
        mtp_draft_input_hiddens=None,
    )

    proposal = proposer.propose_next(
        target_model_input=model_input,
        target_model_output=SimpleNamespace(
            mtp_collector=SimpleNamespace(spec_hidden=target_hidden),
        ),
        target_next_token_ids=model_input.input_ids,
        b_req_mtp_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        draft_step=2,
        accept_len=torch.tensor([2, 2], dtype=torch.int32),
    )

    assert isinstance(proposal, DFlashSpecProposal)
    assert len(forwarded_inputs) == 2
    verify_draft_input, block_draft_input = forwarded_inputs
    assert verify_draft_input is not model_input
    assert verify_draft_input.mtp_draft_input_hiddens is target_hidden
    assert model_input.mtp_draft_input_hiddens is None

    assert torch.equal(block_draft_input.input_ids, torch.tensor([11, 99, 99, 21, 99, 99]))
    assert torch.equal(block_draft_input.b_req_idx, torch.tensor([7, 7, 7, 9, 9, 9], dtype=torch.int32))
    assert torch.equal(block_draft_input.b_mtp_index, torch.zeros(6, dtype=torch.int32))
    assert torch.equal(block_draft_input.b_seq_len, torch.tensor([6, 7, 8, 10, 11, 12], dtype=torch.int32))
    assert torch.equal(block_draft_input.b_position_delta, torch.tensor([1, 1, 1, 2, 2, 2], dtype=torch.int32))
    assert block_draft_input.mtp_draft_input_hiddens is None
    assert block_draft_input.mem_indexes is extra_mem_indexes_cpu
    assert block_draft_input.mem_indexes_cpu is None

    assert torch.equal(proposal.token_ids, torch.tensor([[30, 31], [40, 41]], dtype=torch.int64))
    torch.testing.assert_close(
        proposal.schedule_scores,
        flat_draft_token_probs.reshape(2, block_size)[:, :2].float(),
    )
    assert len(proposal.extra_mem_indexes_cpu) == 1
    assert proposal.extra_mem_indexes_cpu[0].mem_indexes_cpu is extra_mem_indexes_cpu

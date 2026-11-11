from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelMtpOutputCollector
from lightllm.common.basemodel.triton_kernel.mtp_utils import build_dspark_draft_rows

from lightllm.server.router.model_infer.mtp_speculative.proposers.dspark import DSparkProposer
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


@pytest.mark.parametrize("lengths,ready", [([3, 2], None), ([5, 2], [2, 0]), ([], [])])
def test_dspark_prefill_commits_suffix_without_proposal(lengths, ready):
    forwarded_inputs = []
    req_num = len(lengths)
    draft_model = SimpleNamespace(config={"sliding_window": 3}, forward=forwarded_inputs.append)
    proposer = DSparkProposer(
        backend=SimpleNamespace(draft_models=[draft_model], decode_timelog=None, profiler=None),
        enable_dynmaic_mtp=False,
    )
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    starts = seq_lens.cumsum(0, dtype=torch.int32) - seq_lens
    model_input = SimpleNamespace(
        is_prefill=True,
        max_q_seq_len=max(lengths, default=0),
        max_kv_seq_len=max(lengths, default=0),
        max_cache_len=0,
        b_position_delta=None,
        b_req_idx=torch.arange(req_num, dtype=torch.int32) + 3,
        b_seq_len=seq_lens,
        b_ready_cache_len=torch.zeros_like(seq_lens),
        b_prefill_start_loc=starts,
        input_ids=torch.arange(sum(lengths), dtype=torch.int64),
        mtp_draft_input_hiddens=None,
    )
    hidden_rows = sum(lengths) if ready is None else sum(n - r for n, r in zip(lengths, ready))
    target_hidden = torch.empty((hidden_rows, 8))
    proposal = proposer.fill_draft_model_kv_state(
        target_model_input=model_input,
        target_model_output=SimpleNamespace(
            mtp_collector=ModelMtpOutputCollector(
                spec_hidden=target_hidden,
                spec_hidden_prefill_ready_cache_len=None if ready is None else torch.tensor(ready, dtype=torch.int32),
            )
        ),
        target_next_token_ids=torch.arange(req_num, dtype=torch.int64) + 11,
    )
    assert proposal is None
    assert len(forwarded_inputs) == (1 if hidden_rows else 0)
    if not hidden_rows:
        return
    commit = forwarded_inputs[0]
    assert commit is not model_input
    assert commit.mtp_draft_input_hiddens is target_hidden
    assert model_input.mtp_draft_input_hiddens is None
    expected_rows = [
        i for start, n, r in zip(starts.tolist(), lengths, ready or [0] * req_num) for i in range(start + r, start + n)
    ]
    assert commit.input_ids.tolist() == expected_rows
    assert model_input.input_ids.tolist() == list(range(sum(lengths)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_commits_verify_kv_and_builds_parallel_block(monkeypatch):
    block_size = 3
    device = "cuda"
    draft_token_ids = torch.tensor([30, 31, 32, 40, 41, 42], dtype=torch.int64, device=device)
    confidence_logits = torch.tensor(
        [
            [-10.0, 0.0, 10.0],
            [10.0, 0.0, -10.0],
        ],
        device=device,
    )
    block_output = SimpleNamespace(
        mtp_collector=SimpleNamespace(
            draft_token_ids=draft_token_ids,
            confidence_logits=confidence_logits,
        )
    )
    forwarded_inputs = []

    def forward(model_input):
        forwarded_inputs.append(model_input)
        return block_output if len(forwarded_inputs) == 2 else SimpleNamespace()

    draft_model = SimpleNamespace(
        block_size=block_size,
        mask_token_id=99,
        forward=forward,
    )
    proposer = DSparkProposer(
        backend=SimpleNamespace(draft_models=[draft_model], decode_timelog=None, profiler=None),
        enable_dynmaic_mtp=True,
    )
    monkeypatch.setattr(
        g_pin_mem_manager,
        "get_const_gpu_tensor",
        lambda *, shape, fill_value, dtype, **_: torch.full(shape, fill_value, dtype=dtype, device=device),
    )
    monkeypatch.setattr(
        g_pin_mem_manager,
        "async_copy_from_gpu_tensor",
        lambda *, gpu_tensor, **_: gpu_tensor.clone(),
    )

    target_hidden = torch.empty((5, 8), device=device)
    model_input = SimpleNamespace(
        is_prefill=False,
        batch_size=5,
        total_token_num=40,
        max_q_seq_len=1,
        max_kv_seq_len=9,
        input_ids=torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64, device=device),
        b_req_idx=torch.tensor([7, 7, 7, 9, 9], dtype=torch.int32, device=device),
        b_mtp_index=torch.arange(5, dtype=torch.int32, device=device),
        b_seq_len=torch.tensor([4, 5, 6, 8, 9], dtype=torch.int32, device=device),
        b_position_delta=torch.tensor([1, 1, 1, 2, 2], dtype=torch.int32, device=device),
        b_shared_seq_len=torch.tensor([3, 3, 3, 6, 6], dtype=torch.int32, device=device),
        b_shared_radix_node_id=torch.tensor([17, 17, 17, 19, 19], dtype=torch.int64, device=device),
        multimodal_params=[{"images": [], "audios": []} for _ in range(5)],
        mtp_draft_input_hiddens=None,
    )

    proposal = proposer.propose_next(
        target_model_input=model_input,
        target_model_output=SimpleNamespace(
            mtp_collector=SimpleNamespace(spec_hidden=target_hidden),
        ),
        target_next_token_ids=model_input.input_ids,
        b_req_mtp_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
        draft_step=2,
        accept_len=torch.tensor([2, 2], dtype=torch.int32, device=device),
    )

    assert len(forwarded_inputs) == 2
    verify_draft_input, block_draft_input = forwarded_inputs
    assert verify_draft_input is not model_input
    assert verify_draft_input.mtp_draft_input_hiddens is target_hidden
    assert model_input.mtp_draft_input_hiddens is None

    assert block_draft_input.input_ids.tolist() == [11, 99, 99, 21, 99, 99]
    assert block_draft_input.b_req_idx.tolist() == [7, 7, 7, 9, 9, 9]
    assert block_draft_input.b_mtp_index.tolist() == [0] * 6
    assert block_draft_input.b_seq_len.tolist() == [6, 7, 8, 10, 11, 12]
    assert block_draft_input.b_position_delta.tolist() == [1, 1, 1, 2, 2, 2]
    assert block_draft_input.mtp_draft_input_hiddens is None

    assert proposal.token_ids.tolist() == [[30, 31], [40, 41]]
    torch.testing.assert_close(
        proposal.schedule_scores,
        torch.tensor([[0.01, 0.5], [0.99, 0.5]], dtype=torch.float32, device=device),
    )
    assert torch.equal(proposal.schedule_scores_cpu, proposal.schedule_scores)
    assert proposal.schedule_scores_cpu is not proposal.schedule_scores


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("req_num,block_size", [(0, 3), (1, 1), (173, 3), (257, 8)])
def test_dspark_draft_rows_match_target_tails(req_num, block_size):
    rows = torch.arange(req_num * 4, dtype=torch.int32, device="cuda")
    starts = rows[::4].contiguous()
    accepted = starts.remainder(3) + 1
    inputs = (rows.long() + 100, rows, rows + 512, -rows, rows * 2, rows.long() + 2 ** 42)
    mask_token_id = 99
    actual = build_dspark_draft_rows(starts, accepted, *inputs, block_size, mask_token_id)
    tails = (starts + accepted - 1).long()
    tokens = torch.full((req_num, block_size), mask_token_id, device="cuda", dtype=torch.int64)
    tokens[:, 0] = inputs[0][tails]
    expected = [tokens.flatten(), *(value[tails].repeat_interleave(block_size) for value in inputs[1:])]
    expected[2] += torch.arange(1, block_size + 1, device="cuda", dtype=torch.int32).repeat(req_num)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)

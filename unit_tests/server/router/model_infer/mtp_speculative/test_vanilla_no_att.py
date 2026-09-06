from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.select_mtp_rows import select_accepted_tail_rows
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_no_att import (
    DpOverlapVanillaNoAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_no_att import VanillaNoAttProposer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_select_accepted_tail_rows_triton_matches_index_select():
    device = "cuda"
    b_req_mtp_start_loc = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
    accept_len = torch.tensor([2, 3, 2], dtype=torch.int32, device=device)
    expected_rows = torch.tensor([1, 5, 7], dtype=torch.int64, device=device)

    # Exercise non-contiguous source strides as well as hidden widths spanning
    # multiple Triton column blocks.
    input_ids = torch.arange(16, dtype=torch.int64, device=device)[::2]
    hidden = torch.arange(8 * 514, dtype=torch.float32, device=device).reshape(8, 514)[:, ::2]
    b_req_idx = (torch.arange(16, dtype=torch.int32, device=device) + 10)[::2]
    b_mtp_index = torch.arange(8, dtype=torch.int32, device=device)
    b_seq_len = (torch.arange(16, dtype=torch.int32, device=device) + 20)[::2]
    mem_indexes = (torch.arange(16, dtype=torch.int32, device=device) + 100)[::2]
    b_shared_seq_len = (torch.arange(16, dtype=torch.int32, device=device) + 30)[::2]
    b_shared_radix_node_id = (torch.arange(16, dtype=torch.int64, device=device) + 1000)[::2]
    b_position_delta = (torch.arange(16, dtype=torch.int32, device=device) - 8)[::2]

    selected = select_accepted_tail_rows(
        b_req_mtp_start_loc=b_req_mtp_start_loc,
        accept_len=accept_len,
        input_ids=input_ids,
        hidden=hidden,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_seq_len=b_seq_len,
        mem_indexes=mem_indexes,
        b_shared_seq_len=b_shared_seq_len,
        b_shared_radix_node_id=b_shared_radix_node_id,
        b_position_delta=b_position_delta,
    )

    torch.testing.assert_close(selected.input_ids, input_ids.index_select(0, expected_rows))
    torch.testing.assert_close(selected.hidden, hidden.index_select(0, expected_rows))
    torch.testing.assert_close(selected.b_req_idx, b_req_idx.index_select(0, expected_rows))
    torch.testing.assert_close(selected.b_mtp_index, b_mtp_index.index_select(0, expected_rows))
    torch.testing.assert_close(selected.b_seq_len, b_seq_len.index_select(0, expected_rows))
    torch.testing.assert_close(selected.mem_indexes, mem_indexes.index_select(0, expected_rows))
    torch.testing.assert_close(selected.b_shared_seq_len, b_shared_seq_len.index_select(0, expected_rows))
    torch.testing.assert_close(
        selected.b_shared_radix_node_id,
        b_shared_radix_node_id.index_select(0, expected_rows),
    )
    torch.testing.assert_close(selected.b_position_delta, b_position_delta.index_select(0, expected_rows))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vanilla_no_att_runs_all_draft_steps_for_empty_dp_batch():
    device = "cuda"
    draft_batch_sizes = []

    def draft_forward(model_input):
        draft_batch_sizes.append(model_input.batch_size)
        return SimpleNamespace(
            token_ids=torch.empty((0,), dtype=torch.int64, device=device),
            mtp_collector=SimpleNamespace(
                spec_hidden=torch.empty((0, 2), dtype=torch.float32, device=device),
            ),
        )

    backend = SimpleNamespace(
        draft_models=[SimpleNamespace(forward=draft_forward) for _ in range(2)],
        _gen_argmax_token_ids=lambda output: output.token_ids,
    )
    proposer = VanillaNoAttProposer(backend=backend, enable_dynmaic_mtp=False)
    empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
    target_model_input = SimpleNamespace(
        batch_size=0,
        b_req_idx=empty_i32,
        b_mtp_index=empty_i32,
        b_seq_len=empty_i32,
        mem_indexes=empty_i32,
        mem_indexes_cpu=torch.empty((0,), dtype=torch.int32),
        b_position_delta=empty_i32,
        b_shared_seq_len=empty_i32,
        b_shared_radix_node_id=torch.empty((0,), dtype=torch.int64, device=device),
        multimodal_params=[],
    )
    target_model_output = SimpleNamespace(
        mtp_collector=SimpleNamespace(
            spec_hidden=torch.empty((0, 2), dtype=torch.float32, device=device),
        )
    )

    proposal = proposer.propose_next(
        target_model_input=target_model_input,
        target_model_output=target_model_output,
        target_next_token_ids=torch.empty((0,), dtype=torch.int64, device=device),
        b_req_mtp_start_loc=empty_i32,
        draft_step=2,
        accept_len=empty_i32,
    )

    assert draft_batch_sizes == [0, 0]
    assert proposal.token_ids.shape == (0, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vanilla_no_att_proposes_from_one_accepted_tail_per_request():
    device = "cuda"
    draft_calls = []
    draft_outputs = [
        SimpleNamespace(
            token_ids=torch.tensor([21, 24]),
            token_probs=torch.tensor([0.8, 0.7]),
            mtp_collector=SimpleNamespace(spec_hidden=torch.tensor([[102.0, 103.0], [108.0, 109.0]])),
        ),
        SimpleNamespace(
            token_ids=torch.tensor([31, 34]),
            token_probs=torch.tensor([0.6, 0.5]),
            mtp_collector=SimpleNamespace(spec_hidden=torch.tensor([[202.0, 203.0], [208.0, 209.0]])),
        ),
    ]

    def build_draft_model(step):
        def forward(model_input):
            draft_calls.append(
                {
                    "batch_size": model_input.batch_size,
                    "input_ids": model_input.input_ids.cpu(),
                    "draft_hidden": model_input.mtp_draft_input_hiddens.cpu(),
                    "b_req_idx": model_input.b_req_idx.cpu(),
                    "b_mtp_index": model_input.b_mtp_index.cpu(),
                    "b_seq_len": model_input.b_seq_len.cpu(),
                    "mem_indexes": model_input.mem_indexes.cpu(),
                }
            )
            return draft_outputs[step]

        return SimpleNamespace(forward=forward)

    backend = SimpleNamespace(
        draft_models=[build_draft_model(0), build_draft_model(1)],
        _gen_argmax_token_ids=lambda output: output.token_ids,
        _gen_argmax_token_ids_and_prob=lambda output: (output.token_ids, output.token_probs),
    )
    proposer = VanillaNoAttProposer(backend=backend, enable_dynmaic_mtp=True)
    target_model_input = SimpleNamespace(
        batch_size=5,
        input_ids=torch.tensor([10, 11, 12, 13, 14], device=device),
        b_req_idx=torch.tensor([7, 7, 7, 9, 9], dtype=torch.int32, device=device),
        b_mtp_index=torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32, device=device),
        b_seq_len=torch.tensor([10, 11, 12, 20, 21], dtype=torch.int32, device=device),
        mem_indexes=torch.tensor([100, 101, 102, 103, 104], dtype=torch.int32, device=device),
        mem_indexes_cpu=torch.tensor([100, 101, 102, 103, 104], dtype=torch.int32),
        b_position_delta=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device),
        b_shared_seq_len=torch.tensor([8, 8, 8, 6, 6], dtype=torch.int32, device=device),
        b_shared_radix_node_id=torch.tensor([70, 70, 70, 90, 90], dtype=torch.int64, device=device),
        multimodal_params=[{"images": [], "audios": []} for _ in range(5)],
    )
    target_model_output = SimpleNamespace(
        mtp_collector=SimpleNamespace(spec_hidden=torch.arange(10, dtype=torch.float32, device=device).reshape(5, 2))
    )

    proposal = proposer.propose_next(
        target_model_input=target_model_input,
        target_model_output=target_model_output,
        target_next_token_ids=target_model_input.input_ids,
        b_req_mtp_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
        draft_step=2,
        accept_len=torch.tensor([2, 2], dtype=torch.int32, device=device),
    )

    torch.testing.assert_close(proposal.token_ids, torch.tensor([[21, 31], [24, 34]]))
    torch.testing.assert_close(proposal.schedule_scores, torch.tensor([[0.8, 0.6], [0.7, 0.5]]))
    assert len(draft_calls) == 2
    assert draft_calls[0]["batch_size"] == 2
    torch.testing.assert_close(draft_calls[0]["input_ids"], torch.tensor([11, 14]))
    torch.testing.assert_close(draft_calls[0]["draft_hidden"], torch.tensor([[2.0, 3.0], [8.0, 9.0]]))
    torch.testing.assert_close(draft_calls[0]["b_req_idx"], torch.tensor([7, 9], dtype=torch.int32))
    torch.testing.assert_close(draft_calls[0]["b_mtp_index"], torch.tensor([1, 1], dtype=torch.int32))
    torch.testing.assert_close(draft_calls[0]["b_seq_len"], torch.tensor([11, 21], dtype=torch.int32))
    torch.testing.assert_close(draft_calls[0]["mem_indexes"], torch.tensor([101, 104], dtype=torch.int32))
    torch.testing.assert_close(draft_calls[1]["input_ids"], torch.tensor([21, 24]))
    torch.testing.assert_close(
        draft_calls[1]["draft_hidden"],
        torch.tensor([[102.0, 103.0], [108.0, 109.0]]),
    )
    assert target_model_input.batch_size == 5
    torch.testing.assert_close(target_model_input.input_ids.cpu(), torch.tensor([10, 11, 12, 13, 14]))


def test_vanilla_no_att_skips_draft_forward_for_zero_steps():
    proposer = VanillaNoAttProposer(
        backend=SimpleNamespace(draft_models=[]),
        enable_dynmaic_mtp=True,
    )

    proposal = proposer.propose_next(
        target_model_input=None,
        target_model_output=None,
        target_next_token_ids=torch.tensor([10, 11]),
        b_req_mtp_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        draft_step=0,
    )

    assert proposal.token_ids.shape == (2, 0)
    assert proposal.schedule_scores.shape == (2, 0)


def test_vanilla_no_att_fill_hooks_are_noops():
    backend = SimpleNamespace(draft_models=[])
    proposer = VanillaNoAttProposer(backend=backend, enable_dynmaic_mtp=False)
    overlap_proposer = DpOverlapVanillaNoAttProposer(backend=backend, enable_dynmaic_mtp=False)

    proposer.fill_draft_model_kv_state(None, None, None)
    overlap_proposer.fill_draft_model_kv_state_overlap(None, None, None, None, None, None)

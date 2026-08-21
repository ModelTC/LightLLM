from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_with_att import (
    DpOverlapVanillaWithAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.vanilla_with_att import VanillaWithAttProposer


def _prefill_input(input_ids, batch_size):
    return SimpleNamespace(
        is_prefill=True,
        b_position_delta=None,
        batch_size=batch_size,
        input_ids=input_ids,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32, device=input_ids.device),
        b_seq_len=torch.full(
            (batch_size,), input_ids.shape[0] // batch_size, dtype=torch.int32, device=input_ids.device
        ),
        b_ready_cache_len=torch.zeros(batch_size, dtype=torch.int32, device=input_ids.device),
        mtp_draft_input_hiddens=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chained_prefill_advances_local_input_without_mutating_target():
    device = "cuda"
    original_input_ids = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64, device=device)
    target_input = _prefill_input(original_input_ids, batch_size=2)
    target_hidden = torch.arange(12, dtype=torch.float32, device=device).reshape(6, 2)
    forwarded = []

    def draft_model(output_tokens, output_hidden):
        def forward(model_input):
            forwarded.append(
                (
                    model_input,
                    model_input.input_ids.clone(),
                    model_input.mtp_draft_input_hiddens,
                    model_input.b_is_decode_req,
                )
            )
            return SimpleNamespace(
                token_ids=output_tokens,
                mtp_collector=SimpleNamespace(spec_hidden=output_hidden),
            )

        return SimpleNamespace(forward=forward)

    stage0_hidden = target_hidden + 100
    stage1_hidden = target_hidden + 200
    backend = SimpleNamespace(
        draft_models=[
            draft_model(torch.tensor([30, 40], dtype=torch.int64, device=device), stage0_hidden),
            draft_model(torch.tensor([31, 41], dtype=torch.int64, device=device), stage1_hidden),
        ],
        _gen_argmax_token_ids=lambda output: output.token_ids,
    )
    proposer = VanillaWithAttProposer(backend=backend, enable_dynmaic_mtp=False)

    proposer.fill_draft_model_kv_state(
        target_model_input=target_input,
        target_model_output=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=target_hidden)),
        target_next_token_ids=torch.tensor([13, 23], dtype=torch.int64, device=device),
    )

    assert forwarded[0][0] is forwarded[1][0]
    assert forwarded[0][0] is not target_input
    torch.testing.assert_close(
        forwarded[0][1],
        torch.tensor([11, 12, 13, 21, 22, 23], dtype=torch.int64, device=device),
    )
    torch.testing.assert_close(
        forwarded[1][1],
        torch.tensor([12, 13, 30, 22, 23, 40], dtype=torch.int64, device=device),
    )
    assert forwarded[0][2] is target_hidden
    assert forwarded[1][2] is stage0_hidden
    assert not forwarded[0][3].any()
    assert forwarded[0][3].data_ptr() == forwarded[1][3].data_ptr()
    assert target_input.input_ids is original_input_ids
    assert target_input.mtp_draft_input_hiddens is None


def test_overlap_chained_prefill_uses_local_microbatch_inputs(monkeypatch):
    def prepare(model_input, b_next_token_ids, mtp_draft_input_hiddens):
        model_input.input_ids = model_input.input_ids + b_next_token_ids
        model_input.mtp_draft_input_hiddens = mtp_draft_input_hiddens
        return model_input

    target_input0 = _prefill_input(torch.tensor([1, 2], dtype=torch.int64), batch_size=2)
    target_input1 = _prefill_input(torch.tensor([3, 4], dtype=torch.int64), batch_size=2)
    target_hidden0 = torch.tensor([[1.0], [2.0]])
    target_hidden1 = torch.tensor([[3.0], [4.0]])
    forwarded = []

    class DraftModel:
        def __init__(self, token_offset):
            self.token_offset = token_offset

        def _microbatch_overlap_prefill_cuda(self, input0, input1):
            forwarded.append((input0, input1, input0.input_ids.clone(), input1.input_ids.clone()))
            return tuple(
                SimpleNamespace(
                    token_ids=torch.full((2,), self.token_offset + index, dtype=torch.int64),
                    mtp_collector=SimpleNamespace(spec_hidden=hidden + self.token_offset),
                )
                for index, hidden in enumerate((target_hidden0, target_hidden1))
            )

    backend = SimpleNamespace(
        draft_models=[DraftModel(10), DraftModel(20)],
        _gen_argmax_token_ids=lambda output: output.token_ids,
    )
    proposer = DpOverlapVanillaWithAttProposer(backend=backend, enable_dynmaic_mtp=False)
    monkeypatch.setattr(proposer, "_prepare_mtp_prefill_inputs", prepare)

    proposer.fill_draft_model_kv_state_overlap(
        target_model_input0=target_input0,
        target_model_output0=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=target_hidden0)),
        target_next_token_ids0=torch.tensor([5, 6], dtype=torch.int64),
        target_model_input1=target_input1,
        target_model_output1=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=target_hidden1)),
        target_next_token_ids1=torch.tensor([7, 8], dtype=torch.int64),
    )

    assert forwarded[0][0] is forwarded[1][0]
    assert forwarded[0][1] is forwarded[1][1]
    assert forwarded[0][0] is not target_input0
    assert forwarded[0][1] is not target_input1
    torch.testing.assert_close(forwarded[0][2], torch.tensor([6, 8], dtype=torch.int64))
    torch.testing.assert_close(forwarded[0][3], torch.tensor([10, 12], dtype=torch.int64))
    torch.testing.assert_close(forwarded[1][2], torch.tensor([16, 18], dtype=torch.int64))
    torch.testing.assert_close(forwarded[1][3], torch.tensor([21, 23], dtype=torch.int64))
    torch.testing.assert_close(target_input0.input_ids, torch.tensor([1, 2], dtype=torch.int64))
    torch.testing.assert_close(target_input1.input_ids, torch.tensor([3, 4], dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_chained_decode_overlays_verified_tokens_for_the_next_level():
    device = "cuda"
    original_input_ids = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64, device=device)
    target_input = SimpleNamespace(
        is_prefill=False,
        batch_size=6,
        input_ids=original_input_ids,
        mtp_draft_input_hiddens=None,
    )
    target_hidden = torch.arange(12, dtype=torch.float32, device=device).reshape(6, 2)
    forwarded = []

    def draft_model(output_tokens, output_probs, output_hidden):
        def forward(model_input):
            forwarded.append(
                (
                    model_input,
                    model_input.input_ids.clone(),
                    model_input.mtp_draft_input_hiddens,
                )
            )
            return SimpleNamespace(
                token_ids=output_tokens,
                token_probs=output_probs,
                mtp_collector=SimpleNamespace(spec_hidden=output_hidden),
            )

        return SimpleNamespace(forward=forward)

    stage0_tokens = torch.tensor([30, 31, 32, 33, 34, 35], dtype=torch.int64, device=device)
    stage1_tokens = torch.tensor([40, 41, 42, 43, 44, 45], dtype=torch.int64, device=device)
    stage2_tokens = torch.tensor([50, 51, 52, 53, 54, 55], dtype=torch.int64, device=device)
    stage0_probs = torch.tensor([0.30, 0.31, 0.32, 0.33, 0.34, 0.35], device=device)
    stage1_probs = torch.tensor([0.40, 0.41, 0.42, 0.43, 0.44, 0.45], device=device)
    stage2_probs = torch.tensor([0.50, 0.51, 0.52, 0.53, 0.54, 0.55], device=device)
    stage0_hidden = target_hidden + 100
    stage1_hidden = target_hidden + 200
    stage2_hidden = target_hidden + 300
    backend = SimpleNamespace(
        max_draft_step=3,
        draft_models=[
            draft_model(stage0_tokens, stage0_probs, stage0_hidden),
            draft_model(stage1_tokens, stage1_probs, stage1_hidden),
            draft_model(stage2_tokens, stage2_probs, stage2_hidden),
        ],
        _gen_argmax_token_ids_and_prob=lambda output: (output.token_ids, output.token_probs),
    )
    proposer = VanillaWithAttProposer(backend=backend, enable_dynmaic_mtp=True)

    proposal = proposer.propose_next(
        target_model_input=target_input,
        target_model_output=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=target_hidden)),
        target_next_token_ids=original_input_ids,
        b_req_mtp_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
        draft_step=3,
        accept_len=torch.tensor([3, 2], dtype=torch.int32, device=device),
    )

    assert forwarded[0][0] is forwarded[1][0] is forwarded[2][0]
    assert forwarded[0][0] is not target_input
    torch.testing.assert_close(forwarded[0][1], original_input_ids)
    # 第一次覆盖：[10, 11, 12] -> [11, 12, 32]，
    # [20, 21] -> [21, 34]。
    torch.testing.assert_close(
        forwarded[1][1],
        torch.tensor([11, 12, 32, 21, 34, 35], dtype=torch.int64, device=device),
    )
    # 第二次覆盖：[11, 12, 32] -> [12, 32, 42]，
    # [21, 34] -> [34, 44]。
    torch.testing.assert_close(
        forwarded[2][1],
        torch.tensor([12, 32, 42, 34, 44, 45], dtype=torch.int64, device=device),
    )
    assert forwarded[0][2] is target_hidden
    assert forwarded[1][2] is stage0_hidden
    assert forwarded[2][2] is stage1_hidden
    torch.testing.assert_close(
        proposal.token_ids,
        torch.tensor([[32, 42, 52], [34, 44, 54]], dtype=torch.int64, device=device),
    )
    torch.testing.assert_close(
        proposal.schedule_scores,
        torch.tensor([[0.32, 0.42, 0.52], [0.34, 0.44, 0.54]], device=device),
    )
    assert proposal.extra_mem_indexes_cpu == []
    assert target_input.input_ids is original_input_ids
    assert target_input.mtp_draft_input_hiddens is None

    with pytest.raises(AssertionError, match="requires the full chained draft depth"):
        proposer.propose_next(
            target_model_input=target_input,
            target_model_output=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=target_hidden)),
            target_next_token_ids=original_input_ids,
            b_req_mtp_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
            draft_step=2,
            accept_len=torch.tensor([3, 2], dtype=torch.int32, device=device),
        )

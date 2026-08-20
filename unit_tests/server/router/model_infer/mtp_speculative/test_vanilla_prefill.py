from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_utils import (
    fill_dp_chained_mtp_draft_model_kv_state_overlap,
)
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

        def microbatch_overlap_prefill(self, input0, input1):
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

    fill_dp_chained_mtp_draft_model_kv_state_overlap(
        proposer=proposer,
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

from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_no_att import (
    EagleNoAttProposer,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_eagle_no_att_recurrently_proposes_from_accepted_tails():
    device = "cuda"
    draft_calls = []
    draft_outputs = [
        SimpleNamespace(
            token_ids=torch.tensor([21, 24], device=device),
            token_probs=torch.tensor([0.8, 0.7], device=device),
            mtp_collector=SimpleNamespace(spec_hidden=torch.tensor([[102.0, 103.0], [108.0, 109.0]], device=device)),
        ),
        SimpleNamespace(
            token_ids=torch.tensor([31, 34], device=device),
            token_probs=torch.tensor([0.6, 0.5], device=device),
            mtp_collector=SimpleNamespace(spec_hidden=torch.tensor([[202.0, 203.0], [208.0, 209.0]], device=device)),
        ),
    ]

    def forward(model_input):
        draft_calls.append(
            {
                "batch_size": model_input.batch_size,
                "input_ids": model_input.input_ids.cpu(),
                "draft_hidden": model_input.mtp_draft_input_hiddens.cpu(),
                "b_req_idx": model_input.b_req_idx.cpu(),
                "b_seq_len": model_input.b_seq_len.cpu(),
                "mem_indexes": model_input.mem_indexes.cpu(),
            }
        )
        return draft_outputs[len(draft_calls) - 1]

    backend = SimpleNamespace(
        draft_models=[SimpleNamespace(forward=forward)],
        _gen_argmax_token_ids=lambda output: output.token_ids,
        _gen_argmax_token_ids_and_prob=lambda output: (
            output.token_ids,
            output.token_probs,
        ),
    )
    proposer = EagleNoAttProposer(backend=backend, enable_dynmaic_mtp=True)
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

    torch.testing.assert_close(proposal.token_ids, torch.tensor([[21, 31], [24, 34]], device=device))
    torch.testing.assert_close(proposal.schedule_scores, torch.tensor([[0.8, 0.6], [0.7, 0.5]], device=device))
    assert proposal.extra_mem_indexes_cpu == []
    assert len(draft_calls) == 2
    assert draft_calls[0]["batch_size"] == 2
    torch.testing.assert_close(draft_calls[0]["input_ids"], torch.tensor([11, 14]))
    torch.testing.assert_close(draft_calls[0]["draft_hidden"], torch.tensor([[2.0, 3.0], [8.0, 9.0]]))
    torch.testing.assert_close(draft_calls[0]["b_req_idx"], torch.tensor([7, 9], dtype=torch.int32))
    torch.testing.assert_close(draft_calls[0]["b_seq_len"], torch.tensor([11, 21], dtype=torch.int32))
    torch.testing.assert_close(draft_calls[0]["mem_indexes"], torch.tensor([101, 104], dtype=torch.int32))
    torch.testing.assert_close(draft_calls[1]["input_ids"], torch.tensor([21, 24]))
    torch.testing.assert_close(
        draft_calls[1]["draft_hidden"],
        torch.tensor([[102.0, 103.0], [108.0, 109.0]]),
    )
    assert target_model_input.batch_size == 5
    torch.testing.assert_close(target_model_input.input_ids.cpu(), torch.tensor([10, 11, 12, 13, 14]))


def test_eagle_no_att_fill_hook_is_noop():
    proposer = EagleNoAttProposer(backend=SimpleNamespace(draft_models=[]), enable_dynmaic_mtp=False)

    proposer.fill_draft_model_kv_state(None, None, None)

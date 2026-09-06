from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelMtpOutputCollector, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle3 import Eagle3Proposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle_with_att import EagleWithAttProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import EagleSpecProposal


def test_eagle3_reuses_attention_flow_and_maps_proposal_tokens():
    target_input_ids = torch.tensor([10, 20], dtype=torch.int64)
    target_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def forward(model_input):
        assert model_input.input_ids is target_input_ids
        assert model_input.mtp_draft_input_hiddens is target_hidden
        return ModelOutput(
            logits=torch.tensor([[1.0], [2.0]]),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=target_hidden + 10),
        )

    draft_model = SimpleNamespace(
        forward=forward,
        map_draft_vocab_to_main_vocab=lambda token_ids: token_ids + 100,
    )
    proposer = Eagle3Proposer(
        backend=SimpleNamespace(
            draft_models=[draft_model],
            _gen_argmax_token_ids=lambda output: output.logits[:, 0].long(),
        ),
        enable_dynmaic_mtp=False,
    )
    target_input = SimpleNamespace(
        is_prefill=False,
        batch_size=2,
        input_ids=None,
        b_position_delta=torch.zeros(2, dtype=torch.int32),
        mtp_draft_input_hiddens=None,
    )

    proposal = proposer.propose_next(
        target_model_input=target_input,
        target_model_output=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=target_hidden)),
        target_next_token_ids=target_input_ids,
        b_req_mtp_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        draft_step=1,
        accept_len=torch.ones(2, dtype=torch.int32),
    )

    assert isinstance(proposal, EagleSpecProposal)
    torch.testing.assert_close(proposal.token_ids, torch.tensor([[101], [102]]))
    assert proposal.schedule_scores is None
    assert target_input.input_ids is None
    assert target_input.mtp_draft_input_hiddens is None


def test_eagle_with_att_rejects_zero_draft_steps():
    proposer = EagleWithAttProposer(
        backend=SimpleNamespace(draft_models=[]),
        enable_dynmaic_mtp=False,
    )

    with pytest.raises(AssertionError, match="requires draft_step to be greater than 0"):
        proposer.propose_next(
            target_model_input=None,
            target_model_output=None,
            target_next_token_ids=torch.tensor([10]),
            b_req_mtp_start_loc=torch.tensor([0], dtype=torch.int32),
            draft_step=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_eagle_with_att_prefill_builds_draft_kv_without_mutating_target():
    device = "cuda"
    original_input_ids = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64, device=device)
    target_hidden = torch.arange(12, dtype=torch.float32, device=device).reshape(6, 2)
    target_input = SimpleNamespace(
        is_prefill=True,
        batch_size=2,
        input_ids=original_input_ids,
        b_req_idx=torch.tensor([7, 9], dtype=torch.int32, device=device),
        b_seq_len=torch.tensor([3, 3], dtype=torch.int32, device=device),
        b_ready_cache_len=torch.zeros(2, dtype=torch.int32, device=device),
        b_position_delta=None,
        b_is_decode_req=None,
        mtp_draft_input_hiddens=None,
    )
    forwarded = []

    def forward(model_input):
        forwarded.append(
            (
                model_input,
                model_input.input_ids.clone(),
                model_input.mtp_draft_input_hiddens,
                model_input.b_is_decode_req,
            )
        )
        return ModelOutput(logits=torch.empty((0,), device=device))

    proposer = EagleWithAttProposer(
        backend=SimpleNamespace(draft_models=[SimpleNamespace(forward=forward)]),
        enable_dynmaic_mtp=False,
    )
    proposer.fill_draft_model_kv_state(
        target_model_input=target_input,
        target_model_output=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=target_hidden)),
        target_next_token_ids=torch.tensor([13, 23], dtype=torch.int64, device=device),
    )

    assert len(forwarded) == 1
    assert forwarded[0][0] is not target_input
    torch.testing.assert_close(
        forwarded[0][1],
        torch.tensor([11, 12, 13, 21, 22, 23], dtype=torch.int64, device=device),
    )
    assert forwarded[0][2] is target_hidden
    assert not forwarded[0][3].any()
    assert target_input.input_ids is original_input_ids
    assert target_input.b_is_decode_req is None
    assert target_input.mtp_draft_input_hiddens is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_eagle_with_att_commits_verify_kv_then_recurrently_decodes(monkeypatch):
    device = "cuda"
    original_input_ids = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int64, device=device)
    target_hidden = torch.arange(12, dtype=torch.float32, device=device).reshape(6, 2)
    extend_hidden = target_hidden + 100
    decode_hidden = torch.tensor([[201.0, 202.0], [203.0, 204.0]], device=device)
    draft_calls = []

    def forward(model_input):
        draft_calls.append(
            {
                "model_input": model_input,
                "batch_size": model_input.batch_size,
                "input_ids": model_input.input_ids.clone(),
                "draft_hidden": model_input.mtp_draft_input_hiddens.clone(),
                "b_req_idx": model_input.b_req_idx.clone(),
                "b_mtp_index": model_input.b_mtp_index.clone(),
                "b_seq_len": model_input.b_seq_len.clone(),
                "mem_indexes": model_input.mem_indexes.clone(),
                "max_kv_seq_len": model_input.max_kv_seq_len,
                "total_token_num": model_input.total_token_num,
            }
        )
        if len(draft_calls) == 1:
            return ModelOutput(
                logits=torch.arange(30, 36, dtype=torch.float32, device=device).unsqueeze(1),
                mtp_collector=ModelMtpOutputCollector(spec_hidden=extend_hidden),
            )
        return ModelOutput(
            logits=torch.tensor([[40.0], [41.0]], device=device),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=decode_hidden),
        )

    backend = SimpleNamespace(
        draft_models=[SimpleNamespace(forward=forward)],
        _gen_argmax_token_ids=lambda output: output.logits[:, 0].long(),
        _gen_argmax_token_ids_and_prob=lambda output: (
            output.logits[:, 0].long(),
            output.logits[:, 0] / 100,
        ),
    )
    proposer = EagleWithAttProposer(backend=backend, enable_dynmaic_mtp=True)
    target_input = SimpleNamespace(
        is_prefill=False,
        batch_size=6,
        total_token_num=66,
        max_kv_seq_len=22,
        input_ids=original_input_ids,
        b_req_idx=torch.tensor([7, 7, 7, 9, 9, 9], dtype=torch.int32, device=device),
        b_mtp_index=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device=device),
        b_seq_len=torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32, device=device),
        mem_indexes=torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int32, device=device),
        mem_indexes_cpu=torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int32),
        b_position_delta=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=device),
        b_shared_seq_len=torch.tensor([8, 8, 8, 6, 6, 6], dtype=torch.int32, device=device),
        b_shared_radix_node_id=torch.tensor([70, 70, 70, 90, 90, 90], dtype=torch.int64, device=device),
        multimodal_params=[{"images": [], "audios": []} for _ in range(6)],
        mtp_draft_input_hiddens=None,
    )
    extra_mem_indexes_cpu = torch.tensor([200, 201], dtype=torch.int32)
    monkeypatch.setattr(mtp_utils, "alloc_mem_indexes", lambda token_count: extra_mem_indexes_cpu)

    proposal = proposer.propose_next(
        target_model_input=target_input,
        target_model_output=SimpleNamespace(mtp_collector=SimpleNamespace(spec_hidden=target_hidden)),
        target_next_token_ids=original_input_ids,
        b_req_mtp_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
        draft_step=2,
        accept_len=torch.tensor([3, 2], dtype=torch.int32, device=device),
    )

    assert isinstance(proposal, EagleSpecProposal)
    torch.testing.assert_close(proposal.token_ids, torch.tensor([[32, 40], [34, 41]], device=device))
    torch.testing.assert_close(proposal.schedule_scores, torch.tensor([[0.32, 0.40], [0.34, 0.41]], device=device))
    assert len(proposal.extra_mem_indexes_cpu) == 1
    assert proposal.extra_mem_indexes_cpu[0].mem_indexes_cpu is extra_mem_indexes_cpu
    assert len(draft_calls) == 2
    assert draft_calls[0]["model_input"] is not target_input
    assert draft_calls[0]["batch_size"] == 6
    torch.testing.assert_close(draft_calls[0]["input_ids"], original_input_ids)
    torch.testing.assert_close(draft_calls[0]["draft_hidden"], target_hidden)
    torch.testing.assert_close(draft_calls[0]["mem_indexes"], target_input.mem_indexes)
    assert draft_calls[1]["batch_size"] == 2
    torch.testing.assert_close(draft_calls[1]["input_ids"], torch.tensor([32, 34], device=device))
    torch.testing.assert_close(
        draft_calls[1]["draft_hidden"],
        extend_hidden.index_select(0, torch.tensor([2, 4], device=device)),
    )
    torch.testing.assert_close(draft_calls[1]["b_req_idx"], torch.tensor([7, 9], dtype=torch.int32, device=device))
    torch.testing.assert_close(draft_calls[1]["b_mtp_index"], torch.zeros(2, dtype=torch.int32, device=device))
    torch.testing.assert_close(draft_calls[1]["b_seq_len"], torch.tensor([13, 22], dtype=torch.int32, device=device))
    torch.testing.assert_close(draft_calls[1]["mem_indexes"], extra_mem_indexes_cpu.to(device))
    assert draft_calls[1]["max_kv_seq_len"] == 23
    assert draft_calls[1]["total_token_num"] == 46
    assert target_input.input_ids is original_input_ids
    assert target_input.mtp_draft_input_hiddens is None

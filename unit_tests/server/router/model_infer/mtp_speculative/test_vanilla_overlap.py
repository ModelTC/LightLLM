from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelMtpOutputCollector, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_no_att import (
    DpOverlapVanillaNoAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_with_att import (
    DpOverlapVanillaWithAttProposer,
)


class _DraftModel:
    def __init__(self):
        self.decode_batch_sizes = []

    def _microbatch_overlap_decode_cuda(self, input0, input1):
        self.decode_batch_sizes.append((input0.batch_size, input1.batch_size))
        return tuple(
            ModelOutput(
                logits=torch.arange(
                    model_input.batch_size,
                    dtype=torch.float32,
                    device=model_input.input_ids.device,
                ).view(-1, 1),
                mtp_collector=ModelMtpOutputCollector(
                    spec_hidden=torch.ones(
                        (model_input.batch_size, 2),
                        device=model_input.input_ids.device,
                    )
                ),
            )
            for model_input in (input0, input1)
        )


def test_dp_vanilla_no_att_supports_zero_dynamic_draft_step():
    proposer = DpOverlapVanillaNoAttProposer(
        backend=SimpleNamespace(draft_models=[]),
        enable_dynmaic_mtp=True,
    )
    target_next_token_ids0 = torch.tensor([10], dtype=torch.int64)
    target_next_token_ids1 = torch.tensor([11], dtype=torch.int64)

    proposal = proposer.propose_next_overlap(
        target_model_input0=None,
        target_model_output0=None,
        target_next_token_ids0=target_next_token_ids0,
        accept_len0=torch.ones((1,), dtype=torch.int32),
        target_model_input1=None,
        target_model_output1=None,
        target_next_token_ids1=target_next_token_ids1,
        accept_len1=torch.ones((1,), dtype=torch.int32),
        draft_step=0,
    )

    assert proposal.token_ids.shape == (2, 0)
    assert proposal.schedule_scores.shape == (2, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dp_vanilla_proposer_owns_overlap_decode():
    device = "cuda"
    draft_models = [_DraftModel(), _DraftModel()]
    backend = SimpleNamespace(
        max_draft_step=2,
        draft_models=draft_models,
        _gen_argmax_token_ids=lambda output: output.logits[:, 0].to(torch.int64),
    )
    proposer = DpOverlapVanillaWithAttProposer(backend=backend, enable_dynmaic_mtp=False)
    b_mtp_index = torch.arange(3, dtype=torch.int32, device=device)
    model_input0 = SimpleNamespace(
        batch_size=3,
        b_req_idx=torch.arange(3, dtype=torch.int32, device=device),
        b_mtp_index=b_mtp_index,
    )
    model_input1 = SimpleNamespace(
        batch_size=3,
        b_req_idx=torch.arange(3, dtype=torch.int32, device=device),
        b_mtp_index=b_mtp_index,
    )

    proposal = proposer.propose_next_overlap(
        target_model_input0=model_input0,
        target_model_output0=ModelOutput(
            logits=torch.empty((6, 1), device=device),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2), device=device)),
        ),
        target_next_token_ids0=torch.tensor([10, 11, 0], dtype=torch.int64, device=device),
        accept_len0=torch.tensor([2], dtype=torch.int32, device=device),
        target_model_input1=model_input1,
        target_model_output1=ModelOutput(
            logits=torch.empty((6, 1), device=device),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2), device=device)),
        ),
        target_next_token_ids1=torch.tensor([20, 21, 22], dtype=torch.int64, device=device),
        accept_len1=torch.tensor([1], dtype=torch.int32, device=device),
        draft_step=2,
    )

    assert proposal.token_ids.tolist() == [
        [1, 1],
        [0, 0],
    ]
    assert proposal.extra_mem_indexes_cpu == []
    assert draft_models[0].decode_batch_sizes == [(3, 3)]
    assert draft_models[1].decode_batch_sizes == [(3, 3)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dp_vanilla_proposer_builds_padded_inputs_for_empty_verify_rows():
    device = "cuda"
    draft_models = [_DraftModel(), _DraftModel()]
    backend = SimpleNamespace(
        max_draft_step=2,
        draft_models=draft_models,
        _gen_argmax_token_ids=lambda output: output.logits[:, 0].to(torch.int64),
    )
    proposer = DpOverlapVanillaWithAttProposer(backend=backend, enable_dynmaic_mtp=False)
    empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
    model_input0 = SimpleNamespace(batch_size=0, b_req_idx=empty_i32, b_mtp_index=empty_i32)
    model_input1 = SimpleNamespace(batch_size=0, b_req_idx=empty_i32, b_mtp_index=empty_i32)
    model_output0 = ModelOutput(
        logits=torch.empty((0, 1), device=device),
        mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((0, 2), device=device)),
    )
    model_output1 = ModelOutput(
        logits=torch.empty((0, 1), device=device),
        mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((0, 2), device=device)),
    )

    proposal = proposer.propose_next_overlap(
        target_model_input0=model_input0,
        target_model_output0=model_output0,
        target_next_token_ids0=torch.empty((0,), dtype=torch.int64, device=device),
        accept_len0=torch.empty((0,), dtype=torch.int32, device=device),
        target_model_input1=model_input1,
        target_model_output1=model_output1,
        target_next_token_ids1=torch.empty((0,), dtype=torch.int64, device=device),
        accept_len1=torch.empty((0,), dtype=torch.int32, device=device),
        draft_step=2,
    )

    assert proposal.token_ids.shape == (0, 2)
    assert draft_models[0].decode_batch_sizes == [(0, 0)]
    assert draft_models[1].decode_batch_sizes == [(0, 0)]

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelMtpOutputCollector, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.vanilla_with_att import (
    DpOverlapVanillaWithAttProposer,
)


class _DraftModel:
    def __init__(self):
        self.decode_batch_sizes = []

    def microbatch_overlap_decode(self, input0, input1):
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
    model_input0 = SimpleNamespace(batch_size=6)
    model_input1 = SimpleNamespace(batch_size=6)

    proposal = proposer.propose_next_overlap(
        target_model_input0=model_input0,
        target_model_output0=ModelOutput(
            logits=torch.empty((6, 1), device=device),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2), device=device)),
        ),
        target_next_token_ids0=torch.tensor([10, 11, 0, 0, 0, 0], dtype=torch.int64, device=device),
        real_verify_rows0=3,
        accept_len0=torch.tensor([2], dtype=torch.int32, device=device),
        target_model_input1=model_input1,
        target_model_output1=ModelOutput(
            logits=torch.empty((6, 1), device=device),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2), device=device)),
        ),
        target_next_token_ids1=torch.tensor([20, 21, 22, 0, 0, 0], dtype=torch.int64, device=device),
        real_verify_rows1=3,
        accept_len1=torch.tensor([1], dtype=torch.int32, device=device),
        draft_step=2,
    )

    assert proposal.token_ids.tolist() == [
        [1, 1],
        [0, 0],
    ]
    assert proposal.extra_mem_indexes_cpu == []
    assert draft_models[0].decode_batch_sizes == [(6, 6)]
    assert draft_models[1].decode_batch_sizes == [(6, 6)]

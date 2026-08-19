from types import SimpleNamespace

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
                logits=torch.arange(model_input.batch_size, dtype=torch.float32).view(-1, 1),
                mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((model_input.batch_size, 2))),
            )
            for model_input in (input0, input1)
        )


def test_dp_vanilla_proposer_owns_overlap_decode():
    draft_models = [_DraftModel(), _DraftModel()]
    backend = SimpleNamespace(
        draft_models=draft_models,
        _gen_argmax_token_ids=lambda output: output.logits[:, 0].to(torch.int64),
    )
    proposer = DpOverlapVanillaWithAttProposer(backend=backend, enable_dynmaic_mtp=False)
    model_input0 = SimpleNamespace(batch_size=4)
    model_input1 = SimpleNamespace(batch_size=4)

    proposal = proposer.propose_next_overlap(
        main_model_input0=model_input0,
        main_model_output0=ModelOutput(
            logits=torch.empty((4, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((4, 2))),
        ),
        next_token_ids0=torch.tensor([10, 11, 0, 0], dtype=torch.int64),
        real_verify_rows0=2,
        accept_len0=None,
        main_model_input1=model_input1,
        main_model_output1=ModelOutput(
            logits=torch.empty((4, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((4, 2))),
        ),
        next_token_ids1=torch.tensor([20, 21, 22, 0], dtype=torch.int64),
        real_verify_rows1=3,
        accept_len1=None,
        draft_step=2,
    )

    assert proposal.token_ids.tolist() == [
        [10, 0, 0],
        [11, 1, 1],
        [20, 0, 0],
        [21, 1, 1],
        [22, 2, 2],
    ]
    assert proposal.extra_mem_indexes_cpu is None
    assert draft_models[0].decode_batch_sizes == [(4, 4)]
    assert draft_models[1].decode_batch_sizes == [(4, 4)]

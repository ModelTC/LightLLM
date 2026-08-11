from types import SimpleNamespace

import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.server.router.model_infer.speculative.proposers.eagle3 import Eagle3Proposer
from lightllm.server.router.model_infer.speculative.proposers.eagle_mtp import EagleMTPProposer


class _DraftModel:
    def __init__(self):
        self.extend_batch_sizes = None
        self.decode_batch_sizes = []

    def microbatch_overlap_prefill(self, input0, input1):
        self.extend_batch_sizes = (input0.batch_size, input1.batch_size)
        return tuple(
            ModelOutput(
                logits=torch.arange(model_input.batch_size, dtype=torch.float32).view(-1, 1),
                spec_hidden=torch.ones((model_input.batch_size, 2)),
            )
            for model_input in (input0, input1)
        )

    def microbatch_overlap_decode(self, input0, input1):
        self.decode_batch_sizes.append((input0.batch_size, input1.batch_size))
        return tuple(
            ModelOutput(
                logits=torch.arange(model_input.batch_size, dtype=torch.float32).view(-1, 1),
                spec_hidden=torch.ones((model_input.batch_size, 2)),
            )
            for model_input in (input0, input1)
        )


def _target_input(batch_size):
    return SimpleNamespace(
        batch_size=batch_size,
        b_seq_len=torch.arange(batch_size, dtype=torch.int32) + 4,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32),
        b_position_delta=torch.zeros(batch_size, dtype=torch.int32),
        max_kv_seq_len=16,
        max_cache_len=16,
    )


def test_overlap_eagle_extends_verify_rows_then_decodes_logical_batch(monkeypatch):
    # Keep this topology test CPU-only; production allocates the same tensor
    # through the shared CUDA KV manager.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **_: self)
    draft_model = _DraftModel()
    backend = SimpleNamespace(
        max_draft_step=2,
        draft_models=[draft_model],
        model=SimpleNamespace(
            req_manager=SimpleNamespace(
                mem_manager=SimpleNamespace(HOLD_TOKEN_MEMINDEX=99),
            )
        ),
        _gen_argmax_token_ids=lambda output: output.logits[:, 0].to(torch.int64),
    )
    engine = SimpleNamespace(
        backend=backend,
        enable_dynamic_spec=False,
        alloc_extra_mem_indexes=lambda token_count: torch.arange(token_count, dtype=torch.int32),
    )
    proposer = EagleMTPProposer(engine)
    model_input0 = _target_input(batch_size=6)
    model_input1 = _target_input(batch_size=6)

    proposal = proposer.propose_next_overlap(
        main_model_input0=model_input0,
        main_model_output0=ModelOutput(logits=torch.empty((6, 1)), spec_hidden=torch.ones((6, 2))),
        next_token_ids0=torch.arange(6, dtype=torch.int64),
        real_verify_rows0=3,
        accept_len0=torch.tensor([2, 1], dtype=torch.int32),
        main_model_input1=model_input1,
        main_model_output1=ModelOutput(logits=torch.empty((6, 1)), spec_hidden=torch.ones((6, 2))),
        next_token_ids1=torch.arange(10, 16, dtype=torch.int64),
        real_verify_rows1=6,
        accept_len1=torch.tensor([1, 3], dtype=torch.int32),
        draft_step=2,
    )

    assert draft_model.extend_batch_sizes == (6, 6)
    assert draft_model.decode_batch_sizes == [(2, 2)]
    assert proposal.token_ids.shape == (9, 3)
    assert torch.equal(proposal.token_ids[:, 0], torch.tensor([0, 1, 2, 10, 11, 12, 13, 14, 15]))
    assert proposal.extra_mem_indexes_cpu.shape == (3,)


def test_eagle3_maps_draft_token_ids_in_proposer():
    proposer = Eagle3Proposer.__new__(Eagle3Proposer)
    proposer.backend = SimpleNamespace(
        draft_models=[SimpleNamespace(map_draft_vocab_to_main_vocab=lambda token_ids: token_ids + 100)],
        _gen_argmax_token_ids=lambda _: torch.tensor([1, 2]),
        _gen_argmax_token_ids_and_prob=lambda _: (torch.tensor([3, 4]), torch.tensor([0.8, 0.7])),
    )

    token_ids = proposer._gen_argmax_token_ids(ModelOutput(logits=torch.empty(0)))
    token_ids_with_prob, probs = proposer._gen_argmax_token_ids_and_prob(ModelOutput(logits=torch.empty(0)))

    assert torch.equal(token_ids, torch.tensor([101, 102]))
    assert torch.equal(token_ids_with_prob, torch.tensor([103, 104]))
    assert torch.equal(probs, torch.tensor([0.8, 0.7]))

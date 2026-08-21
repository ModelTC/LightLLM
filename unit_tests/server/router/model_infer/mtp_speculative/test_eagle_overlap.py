from types import SimpleNamespace

import torch

from lightllm.common.basemodel.batch_objs import ModelMtpOutputCollector, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle3 import DpOverlapEagle3Proposer
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_with_att import (
    DpOverlapEagleWithAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle3 import Eagle3Proposer


class _DraftModel:
    def __init__(self):
        self.extend_batch_sizes = None
        self.extend_inputs = None
        self.decode_batch_sizes = []
        self.decode_inputs = []

    def _microbatch_overlap_prefill_cuda(self, input0, input1):
        self.extend_batch_sizes = (input0.batch_size, input1.batch_size)
        self.extend_inputs = (input0, input1)
        return tuple(
            ModelOutput(
                logits=torch.arange(model_input.batch_size, dtype=torch.float32).view(-1, 1),
                mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((model_input.batch_size, 2))),
            )
            for model_input in (input0, input1)
        )

    def _microbatch_overlap_decode_cuda(self, input0, input1):
        self.decode_batch_sizes.append((input0.batch_size, input1.batch_size))
        self.decode_inputs.append((input0, input1))
        return tuple(
            ModelOutput(
                logits=torch.arange(model_input.batch_size, dtype=torch.float32).view(-1, 1),
                mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((model_input.batch_size, 2))),
            )
            for model_input in (input0, input1)
        )

    def map_draft_vocab_to_main_vocab(self, token_ids):
        return token_ids


def _target_input(batch_size):
    return SimpleNamespace(
        batch_size=batch_size,
        total_token_num=batch_size,
        input_ids=torch.arange(batch_size, dtype=torch.int64),
        b_seq_len=torch.arange(batch_size, dtype=torch.int32) + 4,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32),
        b_mtp_index=torch.zeros(batch_size, dtype=torch.int32),
        b_position_delta=torch.zeros(batch_size, dtype=torch.int32),
        b_shared_seq_len=torch.zeros(batch_size, dtype=torch.int32),
        b_shared_radix_node_id=torch.arange(batch_size, dtype=torch.int64),
        mem_indexes=torch.arange(batch_size, dtype=torch.int32),
        mem_indexes_cpu=torch.arange(batch_size, dtype=torch.int32),
        max_kv_seq_len=16,
        max_cache_len=16,
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []}] * batch_size,
    )


def test_overlap_eagle_keeps_fixed_verify_layout(monkeypatch):
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
    proposer = DpOverlapEagleWithAttProposer(backend=backend, enable_dynmaic_mtp=False)
    monkeypatch.setattr(
        mtp_utils,
        "alloc_mem_indexes",
        lambda token_count: torch.arange(token_count, dtype=torch.int32),
    )
    model_input0 = _target_input(batch_size=6)
    model_input1 = _target_input(batch_size=6)

    proposal = proposer.propose_next_overlap(
        target_model_input0=model_input0,
        target_model_output0=ModelOutput(
            logits=torch.empty((6, 1)), mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2)))
        ),
        target_next_token_ids0=torch.arange(6, dtype=torch.int64),
        real_verify_rows0=3,
        accept_len0=torch.tensor([2, 1], dtype=torch.int32),
        target_model_input1=model_input1,
        target_model_output1=ModelOutput(
            logits=torch.empty((6, 1)), mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2)))
        ),
        target_next_token_ids1=torch.arange(10, 16, dtype=torch.int64),
        real_verify_rows1=6,
        accept_len1=torch.tensor([1, 3], dtype=torch.int32),
        draft_step=2,
    )

    assert draft_model.extend_batch_sizes is None
    assert draft_model.decode_batch_sizes == [(6, 6), (6, 6)]
    assert proposal.token_ids.shape == (3, 2)
    expected_draft_tokens = torch.tensor([1, 0, 5])
    assert torch.equal(proposal.token_ids[:, 0], expected_draft_tokens)
    assert torch.equal(proposal.token_ids[:, 1], expected_draft_tokens)
    assert len(proposal.extra_mem_indexes_cpu) == 1
    assert torch.equal(proposal.extra_mem_indexes_cpu[0].mem_indexes_cpu, torch.arange(6, dtype=torch.int32))
    assert proposal.extra_mem_indexes_cpu[0].free_mask_cpu is None
    assert torch.equal(model_input0.mem_indexes, torch.tensor([2, 0, 1, 5, 99, 99], dtype=torch.int32))
    assert torch.equal(model_input1.mem_indexes, torch.tensor([2, 2, 4, 5, 3, 5], dtype=torch.int32))


def test_autoregressive_eagle_reuses_overlap_inputs(monkeypatch):
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
    proposer = DpOverlapEagle3Proposer(backend=backend, enable_dynmaic_mtp=False)
    monkeypatch.setattr(
        mtp_utils,
        "alloc_mem_indexes",
        lambda token_count: torch.arange(token_count, dtype=torch.int32),
    )
    model_input0 = _target_input(batch_size=6)
    model_input1 = _target_input(batch_size=6)

    proposal = proposer.propose_next_overlap(
        target_model_input0=model_input0,
        target_model_output0=ModelOutput(
            logits=torch.empty((6, 1)), mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2)))
        ),
        target_next_token_ids0=torch.arange(6, dtype=torch.int64),
        real_verify_rows0=3,
        accept_len0=torch.tensor([2, 1], dtype=torch.int32),
        target_model_input1=model_input1,
        target_model_output1=ModelOutput(
            logits=torch.empty((6, 1)), mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2)))
        ),
        target_next_token_ids1=torch.arange(10, 16, dtype=torch.int64),
        real_verify_rows1=6,
        accept_len1=torch.tensor([1, 3], dtype=torch.int32),
        draft_step=2,
    )

    assert draft_model.extend_inputs is None
    assert len(draft_model.decode_inputs) == 2
    assert draft_model.decode_inputs[0][0] is model_input0
    assert draft_model.decode_inputs[0][1] is model_input1
    assert draft_model.extend_batch_sizes is None
    assert draft_model.decode_batch_sizes == [(6, 6), (2, 2)]
    assert proposal.token_ids.shape == (3, 2)
    assert torch.equal(proposal.token_ids, torch.tensor([[1, 0], [0, 0], [5, 1]]))
    assert len(proposal.extra_mem_indexes_cpu) == 1
    assert torch.equal(proposal.extra_mem_indexes_cpu[0].mem_indexes_cpu, torch.arange(3, dtype=torch.int32))
    assert proposal.extra_mem_indexes_cpu[0].free_mask_cpu is None


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

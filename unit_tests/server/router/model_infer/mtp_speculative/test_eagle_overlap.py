from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelInput, ModelMtpOutputCollector, ModelOutput
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_engine import DPOverlapSpecEngine
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers import eagle_with_att
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.utils import (
    get_dp_overlap_req_start_rows,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle3 import (
    DpOverlapEagle3Proposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_no_att import (
    DpOverlapEagleNoAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.dp_overlap_proposers.eagle_with_att import (
    DpOverlapEagleWithAttProposer,
)
from lightllm.server.router.model_infer.mtp_speculative.proposers.eagle3 import (
    Eagle3Proposer,
)
from lightllm.server.router.model_infer.pin_mem_manager import AsyncPinnedCpuTensor


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

    def map_draft_vocab_to_main_vocab(self, token_ids):
        return token_ids


def _target_input(batch_size, b_mtp_index=None, device="cpu"):
    if b_mtp_index is None:
        b_mtp_index = torch.arange(batch_size, dtype=torch.int32, device=device) % 3
    return SimpleNamespace(
        batch_size=batch_size,
        total_token_num=batch_size,
        input_ids=torch.arange(batch_size, dtype=torch.int64, device=device),
        b_seq_len=torch.arange(batch_size, dtype=torch.int32, device=device) + 4,
        b_req_idx=torch.arange(batch_size, dtype=torch.int32, device=device),
        b_mtp_index=b_mtp_index,
        b_position_delta=torch.zeros(batch_size, dtype=torch.int32, device=device),
        b_shared_seq_len=torch.zeros(batch_size, dtype=torch.int32, device=device),
        b_shared_radix_node_id=torch.arange(batch_size, dtype=torch.int64, device=device),
        max_kv_seq_len=16,
        max_cache_len=16,
        is_prefill=False,
        multimodal_params=[{"images": [], "audios": []}] * batch_size,
    )


def _patch_cpu_req_start_rows(monkeypatch):
    def get_cpu_req_start_rows(b_mtp_index, req_num):
        req_start_rows = torch.nonzero(b_mtp_index == 0, as_tuple=False).flatten().to(dtype=torch.int32)
        assert req_start_rows.shape == (req_num,)
        return req_start_rows

    monkeypatch.setattr(eagle_with_att, "get_dp_overlap_req_start_rows", get_cpu_req_start_rows)


def test_dp_overlap_req_start_rows_rejects_nonempty_cpu_input():
    with pytest.raises(AssertionError, match="must be a CUDA tensor"):
        get_dp_overlap_req_start_rows(
            b_mtp_index=torch.tensor([0, 1], dtype=torch.int32),
            req_num=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("draft_step", [1, 3])
@pytest.mark.parametrize("empty_batches", [(False, False), (True, False), (False, True), (True, True)])
def test_dsv4_overlap_eagle_uses_one_cpu_accept_buffer(draft_step, empty_batches):
    layouts = ([0, 1, 0, 1, 2], [0, 1])
    accept_lengths = ([2, 1], [1])
    model_inputs, accepts, expected_tails = [], [], []
    for empty, layout, lengths in zip(empty_batches, layouts, accept_lengths):
        mtp_index = torch.tensor([] if empty else layout, dtype=torch.int32)
        model_input = ModelInput(**vars(_target_input(mtp_index.numel(), mtp_index)), max_q_seq_len=1)
        cpu_accept = torch.tensor([] if empty else lengths, dtype=torch.int32)
        start_rows = torch.nonzero(mtp_index == 0, as_tuple=False).flatten()
        expected_tails.append(model_input.b_seq_len.index_select(0, start_rows + cpu_accept - 1).long())
        model_input.to_cuda()
        model_inputs.append(model_input)
        accepts.append(cpu_accept.cuda())

    calls = []

    def forward(input0, input1):
        outputs = []
        for model_input in (input0, input1):
            assert torch.equal(model_input.b_req_idx_cpu, model_input.b_req_idx.cpu())
            assert torch.equal(model_input.b_mtp_index_cpu, model_input.b_mtp_index.cpu())
            assert torch.equal(model_input.b_seq_len_cpu, model_input.b_seq_len.cpu())
            outputs.append(
                ModelOutput(
                    logits=model_input.b_seq_len.float().unsqueeze(1),
                    mtp_collector=ModelMtpOutputCollector(
                        spec_hidden=torch.ones((model_input.batch_size, 2), device="cuda")
                    ),
                )
            )
        calls.append(True)
        return tuple(outputs)

    waits = []
    combined_cpu = torch.zeros(sum(accept.numel() for accept in accepts), dtype=torch.int32)

    def synchronize():
        # Sentinel contents must not be read/split into accepted tails before the wait.
        waits.append(True)
        combined_cpu.copy_(torch.cat(accepts).cpu())

    backend = SimpleNamespace(
        is_deepseek_v4=True,
        draft_models=[SimpleNamespace(_microbatch_overlap_decode_cuda=forward)],
        _gen_argmax_token_ids=lambda output: output.logits[:, 0].long(),
    )
    engine = DPOverlapSpecEngine.__new__(DPOverlapSpecEngine)
    engine.proposer = DpOverlapEagleWithAttProposer(backend=backend, enable_dynmaic_mtp=False)
    outputs = [
        ModelOutput(
            logits=torch.empty((model_input.batch_size, 1), device="cuda"),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((model_input.batch_size, 2), device="cuda")),
        )
        for model_input in model_inputs
    ]
    proposal = engine.propose_next_overlap(
        target_model_input0=model_inputs[0],
        target_model_output0=outputs[0],
        target_next_token_ids0=model_inputs[0].input_ids,
        accept_len0=accepts[0],
        target_model_input1=model_inputs[1],
        target_model_output1=outputs[1],
        target_next_token_ids1=model_inputs[1].input_ids,
        accept_len1=accepts[1],
        draft_step=draft_step,
        accept_len_cpu=AsyncPinnedCpuTensor(combined_cpu, SimpleNamespace(synchronize=synchronize))
        if combined_cpu.numel()
        else None,
    )
    expected = torch.cat(expected_tails)[:, None] + torch.arange(draft_step)[None, :]
    torch.testing.assert_close(proposal.token_ids.cpu(), expected)
    assert len(calls) == draft_step
    assert len(waits) == int(combined_cpu.numel() > 0 and draft_step > 1)
    for model_input in model_inputs:
        torch.testing.assert_close(model_input.b_seq_len.cpu(), model_input.b_seq_len_cpu)


def test_overlap_eagle_supports_variable_verify_layout(monkeypatch):
    _patch_cpu_req_start_rows(monkeypatch)
    draft_model = _DraftModel()
    backend = SimpleNamespace(
        is_deepseek_v4=False,
        max_draft_step=2,
        draft_models=[draft_model],
        model=SimpleNamespace(
            req_manager=SimpleNamespace(
                mem_manager=SimpleNamespace(HOLD_TOKEN_MEMINDEXES=(99,), page_size=1),
            )
        ),
        _gen_argmax_token_ids=lambda output: output.logits[:, 0].to(torch.int64),
    )
    proposer = DpOverlapEagleWithAttProposer(backend=backend, enable_dynmaic_mtp=False)
    model_input0 = _target_input(batch_size=3)
    model_input1 = _target_input(batch_size=6)

    proposal = proposer.propose_next_overlap(
        target_model_input0=model_input0,
        target_model_output0=ModelOutput(
            logits=torch.empty((3, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((3, 2))),
        ),
        target_next_token_ids0=torch.arange(3, dtype=torch.int64),
        accept_len0=torch.tensor([2], dtype=torch.int32),
        target_model_input1=model_input1,
        target_model_output1=ModelOutput(
            logits=torch.empty((6, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2))),
        ),
        target_next_token_ids1=torch.arange(10, 16, dtype=torch.int64),
        accept_len1=torch.tensor([1, 3], dtype=torch.int32),
        draft_step=2,
    )

    assert draft_model.extend_batch_sizes is None
    assert draft_model.decode_batch_sizes == [(3, 6), (1, 2)]
    assert proposal.token_ids.shape == (3, 2)
    assert torch.equal(proposal.token_ids, torch.tensor([[1, 0], [0, 0], [5, 1]]))


def test_overlap_eagle_supports_empty_verify_rows():
    draft_model = _DraftModel()
    backend = SimpleNamespace(
        is_deepseek_v4=False,
        max_draft_step=2,
        draft_models=[draft_model],
        model=SimpleNamespace(
            req_manager=SimpleNamespace(
                mem_manager=SimpleNamespace(HOLD_TOKEN_MEMINDEXES=(99,), page_size=1),
            )
        ),
        _gen_argmax_token_ids=lambda output: output.logits[:, 0].to(torch.int64),
    )
    proposer = DpOverlapEagleWithAttProposer(backend=backend, enable_dynmaic_mtp=False)

    proposal = proposer.propose_next_overlap(
        target_model_input0=_target_input(batch_size=0),
        target_model_output0=ModelOutput(
            logits=torch.empty((0, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((0, 2))),
        ),
        target_next_token_ids0=torch.empty((0,), dtype=torch.int64),
        accept_len0=torch.empty((0,), dtype=torch.int32),
        target_model_input1=_target_input(batch_size=0),
        target_model_output1=ModelOutput(
            logits=torch.empty((0, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((0, 2))),
        ),
        target_next_token_ids1=torch.empty((0,), dtype=torch.int64),
        accept_len1=torch.empty((0,), dtype=torch.int32),
        draft_step=2,
    )

    assert proposal.token_ids.shape == (0, 2)
    assert draft_model.decode_batch_sizes == [(0, 0), (0, 0)]


def test_overlap_eagle_returns_dynamic_schedule_scores(monkeypatch):
    _patch_cpu_req_start_rows(monkeypatch)
    draft_model = _DraftModel()
    backend = SimpleNamespace(
        is_deepseek_v4=False,
        max_draft_step=2,
        draft_models=[draft_model],
        model=SimpleNamespace(
            req_manager=SimpleNamespace(
                mem_manager=SimpleNamespace(HOLD_TOKEN_MEMINDEXES=(99,), page_size=1),
            )
        ),
        _gen_argmax_token_ids_and_prob=lambda output: (
            output.logits[:, 0].to(torch.int64),
            output.logits[:, 0] / 10 + 0.5,
        ),
    )
    proposer = DpOverlapEagleWithAttProposer(backend=backend, enable_dynmaic_mtp=True)

    proposal = proposer.propose_next_overlap(
        target_model_input0=_target_input(
            batch_size=2,
            b_mtp_index=torch.tensor([0, 1], dtype=torch.int32),
        ),
        target_model_output0=ModelOutput(
            logits=torch.empty((2, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((2, 2))),
        ),
        target_next_token_ids0=torch.arange(2, dtype=torch.int64),
        accept_len0=torch.tensor([2], dtype=torch.int32),
        target_model_input1=_target_input(
            batch_size=3,
            b_mtp_index=torch.tensor([0, 1, 0], dtype=torch.int32),
        ),
        target_model_output1=ModelOutput(
            logits=torch.empty((3, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((3, 2))),
        ),
        target_next_token_ids1=torch.arange(2, 5, dtype=torch.int64),
        accept_len1=torch.tensor([1, 1], dtype=torch.int32),
        draft_step=2,
    )

    assert torch.equal(proposal.token_ids, torch.tensor([[1, 0], [0, 0], [2, 1]]))
    assert torch.allclose(
        proposal.schedule_scores,
        torch.tensor([[0.6, 0.5], [0.5, 0.5], [0.7, 0.6]]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_overlap_eagle_no_att_supports_dynamic_draft_step():
    device = "cuda"
    draft_model = _DraftModel()
    backend = SimpleNamespace(
        is_deepseek_v4=False,
        max_draft_step=3,
        draft_models=[draft_model],
        _gen_argmax_token_ids_and_prob=lambda output: (
            output.logits[:, 0].to(torch.int64),
            output.logits[:, 0] / 10 + 0.5,
        ),
    )
    proposer = DpOverlapEagleNoAttProposer(backend=backend, enable_dynmaic_mtp=True)
    model_input0 = _target_input(
        batch_size=2,
        b_mtp_index=torch.tensor([0, 1], dtype=torch.int32, device=device),
        device=device,
    )
    model_input1 = _target_input(
        batch_size=3,
        b_mtp_index=torch.tensor([0, 1, 0], dtype=torch.int32, device=device),
        device=device,
    )

    proposal = proposer.propose_next_overlap(
        target_model_input0=model_input0,
        target_model_output0=ModelOutput(
            logits=torch.empty((2, 1), device=device),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((2, 2), device=device)),
        ),
        target_next_token_ids0=torch.arange(2, dtype=torch.int64, device=device),
        accept_len0=torch.tensor([2], dtype=torch.int32, device=device),
        target_model_input1=model_input1,
        target_model_output1=ModelOutput(
            logits=torch.empty((3, 1), device=device),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((3, 2), device=device)),
        ),
        target_next_token_ids1=torch.arange(2, 5, dtype=torch.int64, device=device),
        accept_len1=torch.tensor([1, 1], dtype=torch.int32, device=device),
        draft_step=2,
    )

    assert proposal.token_ids.tolist() == [[0, 0], [0, 0], [1, 1]]
    assert torch.allclose(
        proposal.schedule_scores,
        torch.tensor(
            [[0.5, 0.5], [0.5, 0.5], [0.6, 0.6]],
            device=device,
        ),
    )
    assert draft_model.decode_batch_sizes == [(1, 2), (1, 2)]


def test_autoregressive_eagle_reuses_overlap_inputs(monkeypatch):
    _patch_cpu_req_start_rows(monkeypatch)
    draft_model = _DraftModel()
    backend = SimpleNamespace(
        is_deepseek_v4=False,
        max_draft_step=2,
        draft_models=[draft_model],
        model=SimpleNamespace(
            req_manager=SimpleNamespace(
                mem_manager=SimpleNamespace(HOLD_TOKEN_MEMINDEXES=(99,), page_size=1),
            )
        ),
        _gen_argmax_token_ids=lambda output: output.logits[:, 0].to(torch.int64),
    )
    proposer = DpOverlapEagle3Proposer(backend=backend, enable_dynmaic_mtp=False)
    model_input0 = _target_input(batch_size=3)
    model_input1 = _target_input(batch_size=6)

    proposal = proposer.propose_next_overlap(
        target_model_input0=model_input0,
        target_model_output0=ModelOutput(
            logits=torch.empty((3, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((3, 2))),
        ),
        target_next_token_ids0=torch.arange(3, dtype=torch.int64),
        accept_len0=torch.tensor([2], dtype=torch.int32),
        target_model_input1=model_input1,
        target_model_output1=ModelOutput(
            logits=torch.empty((6, 1)),
            mtp_collector=ModelMtpOutputCollector(spec_hidden=torch.ones((6, 2))),
        ),
        target_next_token_ids1=torch.arange(10, 16, dtype=torch.int64),
        accept_len1=torch.tensor([1, 3], dtype=torch.int32),
        draft_step=2,
    )

    assert draft_model.extend_inputs is None
    assert len(draft_model.decode_inputs) == 2
    assert draft_model.decode_inputs[0][0] is not model_input0
    assert draft_model.decode_inputs[0][1] is not model_input1
    assert draft_model.extend_batch_sizes is None
    assert draft_model.decode_batch_sizes == [(3, 6), (1, 2)]
    assert proposal.token_ids.shape == (3, 2)
    assert torch.equal(proposal.token_ids, torch.tensor([[1, 0], [0, 0], [5, 1]]))


def test_eagle3_maps_draft_token_ids_in_proposer():
    proposer = Eagle3Proposer.__new__(Eagle3Proposer)
    proposer.backend = SimpleNamespace(
        is_deepseek_v4=False,
        draft_models=[SimpleNamespace(map_draft_vocab_to_main_vocab=lambda token_ids: token_ids + 100)],
        _gen_argmax_token_ids=lambda _: torch.tensor([1, 2]),
        _gen_argmax_token_ids_and_prob=lambda _: (
            torch.tensor([3, 4]),
            torch.tensor([0.8, 0.7]),
        ),
    )

    token_ids = proposer._gen_argmax_token_ids(ModelOutput(logits=torch.empty(0)))
    token_ids_with_prob, probs = proposer._gen_argmax_token_ids_and_prob(ModelOutput(logits=torch.empty(0)))

    assert torch.equal(token_ids, torch.tensor([101, 102]))
    assert torch.equal(token_ids_with_prob, torch.tensor([103, 104]))
    assert torch.equal(probs, torch.tensor([0.8, 0.7]))


def test_dp_overlap_eagle3_maps_draft_token_ids_in_proposer():
    proposer = DpOverlapEagle3Proposer.__new__(DpOverlapEagle3Proposer)
    proposer.backend = SimpleNamespace(
        is_deepseek_v4=False,
        draft_models=[SimpleNamespace(map_draft_vocab_to_main_vocab=lambda token_ids: token_ids + 100)],
        _gen_argmax_token_ids=lambda _: torch.tensor([1, 2]),
        _gen_argmax_token_ids_and_prob=lambda _: (
            torch.tensor([3, 4]),
            torch.tensor([0.8, 0.7]),
        ),
    )

    token_ids = proposer._gen_argmax_token_ids(ModelOutput(logits=torch.empty(0)))
    token_ids_with_prob, probs = proposer._gen_argmax_token_ids_and_prob(ModelOutput(logits=torch.empty(0)))

    assert torch.equal(token_ids, torch.tensor([101, 102]))
    assert torch.equal(token_ids_with_prob, torch.tensor([103, 104]))
    assert torch.equal(probs, torch.tensor([0.8, 0.7]))

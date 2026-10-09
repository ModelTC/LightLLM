from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.batch_objs import ModelMtpOutputCollector, ModelOutput, PostLayerOutput
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import DFlashSpecProposal
from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash2 import DFlash2Proposer


@pytest.mark.parametrize("dynamic_verify", [False, True])
def test_dflash2_proposal_returns_greedy_tokens(dynamic_verify):
    tokens = torch.arange(6).view(2, 3)
    proposer = DFlash2Proposer(backend=SimpleNamespace(), enable_dynmaic_mtp=dynamic_verify)
    confidence = torch.tensor([[0.0, 1.0, -1.0], [float("inf"), -100.0, 2.0]])
    output = SimpleNamespace(
        mtp_collector=ModelMtpOutputCollector(draft_token_ids=tokens, confidence_logits=confidence)
    )
    mem = torch.tensor([7, 8], dtype=torch.int32)
    proposal = proposer._build_proposal(output, 2, 4, 3, mem)
    assert type(proposal) is DFlashSpecProposal
    torch.testing.assert_close(proposal.token_ids, tokens)
    torch.testing.assert_close(proposal.extra_mem_indexes_cpu[0].mem_indexes_cpu, mem)
    if dynamic_verify:
        torch.testing.assert_close(proposal.schedule_scores, confidence.sigmoid().clamp(0.01, 0.99))
    else:
        assert proposal.schedule_scores is None


def test_dflash2_collector_unpads_logical_request_rows():
    tokens = torch.arange(6).view(2, 3)
    collector = ModelMtpOutputCollector(draft_token_ids=tokens, confidence_logits=tokens.float())
    unpadded = collector.unpad_decode(padded_batch_size=8, origin_batch_size=4)
    torch.testing.assert_close(unpadded.draft_token_ids, tokens[:1])
    torch.testing.assert_close(unpadded.confidence_logits, tokens[:1].float())
    assert collector.draft_token_ids.shape[0] == collector.confidence_logits.shape[0] == 2


def test_dflash2_dynamic_verification_requires_confidence():
    proposer = DFlash2Proposer(backend=SimpleNamespace(), enable_dynmaic_mtp=True)
    output = ModelOutput(
        logits=torch.empty(4, 1), mtp_collector=ModelMtpOutputCollector(draft_token_ids=torch.ones(1, 3))
    )
    with pytest.raises(RuntimeError, match="selector confidence"):
        proposer._build_proposal(output, 1, 4, 3, torch.arange(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [1, 3, 7])
@pytest.mark.parametrize("dynamic_verify", [False, True])
def test_greedy_selector_conditions_on_selected_predecessor(width, dynamic_verify):
    from lightllm.models.qwen3_dflash2.triton_kernel.selector_walk import greedy_selector_walk

    scores = torch.randn(3, width, 16, 16, generator=torch.Generator().manual_seed(42))
    scores[0].zero_()  # Ties select the first candidate.
    candidates = torch.arange(3 * width * 16).view(3, width, 16)
    expected = torch.empty(3, width, dtype=torch.int64)
    expected_probs = torch.empty(3, width)
    for req in range(3):
        previous = 0
        for slot in range(width):
            row = scores[req, slot, previous]
            previous = int(row.argmax())
            expected_probs[req, slot] = row.softmax(-1)[previous]
            expected[req, slot] = candidates[req, slot, previous]
    scores, candidates = scores.cuda(), candidates.cuda()
    rng_before = torch.cuda.get_rng_state()
    confidence = torch.empty(3, width, device="cuda") if dynamic_verify else None
    actual = greedy_selector_walk(scores, candidates, confidence_logits=confidence)
    if dynamic_verify:
        torch.testing.assert_close(confidence.sigmoid().cpu(), expected_probs)
    torch.testing.assert_close(actual.cpu(), expected)
    assert torch.equal(rng_before, torch.cuda.get_rng_state())


@pytest.mark.parametrize("is_prefill", [True, False])
def test_dflash2_post_layer_satisfies_base_model_output_contract(is_prefill):
    from lightllm.models.qwen3_dflash2.layer_infer.post_layer_infer import Qwen3DFlash2PostLayerInfer

    layer = Qwen3DFlash2PostLayerInfer.__new__(Qwen3DFlash2PostLayerInfer)
    layer.block_size_ = 4
    layer.selector_top_k_ = 2
    hidden = torch.randn(4, 8)
    ids = torch.arange(6).view(3, 2)
    unary = torch.randn(3, 2)
    selected = ids[:, 0].view(1, 3)
    layer._slice_get_last_input = lambda *_: (hidden, 4)
    layer._norm = lambda x, *_: x
    layer._compute_candidates = lambda **_: (ids, unary)
    confidence = torch.randn(1, 3)
    layer._select_path = lambda **_: (selected, confidence)
    collected = {}
    state = SimpleNamespace(
        is_prefill=is_prefill,
        input_ids=torch.arange(4),
        hidden_collector=SimpleNamespace(add_mtp_outputs=lambda **kwargs: collected.update(kwargs)),
    )
    output = layer.token_forward(hidden, state, None)
    assert isinstance(output, PostLayerOutput)
    assert output.logits.shape == ((0,) if is_prefill else (4, 1))
    if not is_prefill:
        torch.testing.assert_close(collected["draft_token_ids"], selected)
        torch.testing.assert_close(collected["confidence_logits"], confidence)

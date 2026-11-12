from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen4_exp.lexical_state import Qwen4ExpLexicalState
from lightllm.models.qwen4_exp.speculative_cache import LexicalSpeculativeCache
from lightllm.server.router.model_infer.mtp_speculative.utils import verify_mtp_tokens
from .lexical_helpers import make_cache, metadata, reference_segment
from .reference import reference_step
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.qwen4_exp.layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("capture", [False, True])
def test_actual_verifier_commits_only_accepted_lexical_candidates(config, capture):
    torch.manual_seed(1907)
    cache, source = make_cache(config)
    cache.ngram[2].copy_(torch.tensor([7, 11], device="cuda"))
    cache.ngram[5].copy_(torch.tensor([13, 17], device="cuda"))
    cache.convolution.normal_()
    untouched = cache.snapshot(7)
    speculative = LexicalSpeculativeCache(cache, 4)
    req = SimpleNamespace()
    req.speculative_lexical_caches = {1: speculative}
    req.req_sampling_params_manager = SimpleNamespace(
        req_to_next_token_ids=torch.full((8, 4), -1, device="cuda", dtype=torch.int64)
    )
    req.req_to_mtp_state_index = torch.zeros(8, device="cuda", dtype=torch.int32)
    req.HOLD_REQUEST_ID = 7
    lexical = Qwen4ExpLexicalState.__new__(Qwen4ExpLexicalState)
    lexical.request_capacity = 8
    lexical.speculative_lexical_caches = {1: speculative}

    def commit(requests, accepted, req_verify_width, role):
        assert role == "main"
        lexical.commit_bounded_att_state(requests, accepted, req_verify_width)
        req.req_to_mtp_state_index[requests] = accepted - 1

    backend = SimpleNamespace(
        model=SimpleNamespace(req_manager=req, att_state_manager=SimpleNamespace(commit_bounded_att_state=commit)),
        is_linear_att_mixed_model=True,
        max_draft_step=3,
    )
    meta = metadata([5, 5, 5, 2, 2, 7], [11, 12, 13, 11, 12, 1], [1] * 6)
    tokens = torch.empty(6, device="cuda", dtype=torch.int64)
    residual = torch.empty(6, 64, device="cuda", dtype=torch.bfloat16)

    def run():
        batch = cache.prepare(meta)
        result = batch.apply(residual, tokens, cache.layer_weight, 1, None)
        speculative.stage(meta, batch)
        return result

    tokens.fill_(19)
    residual.normal_()
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()

    cases = (
        ([5, 5, 5, 2, 2, 7], [2, 1], [19, 63, 23, 29, 31, 37]),
        ([2, 2, 5, 5, 5, 7], [2, 3], [41, 43, 47, 63, 5, 7]),
        ([5, 5, 5, 5, 2, 7], [1, 1], [11, 13, 17, 19, 23, 29]),
    )
    for requests, accepts, ids in cases:
        starts = [0] + [i for i in range(1, len(requests)) if requests[i] != requests[i - 1]]
        counts = [end - start for start, end in zip(starts, starts[1:] + [len(requests)])]
        lengths = []
        for count in counts:
            lengths.extend(range(11, 11 + count))
        meta.copy_from(metadata(requests, lengths, [1] * 6))
        tokens.copy_(torch.tensor(ids, device="cuda"))
        residual.normal_()
        before_ngram, before_conv = cache.ngram.clone(), cache.convolution.clone()
        if capture:
            graph.replay()
            actual = result
        else:
            actual = run()
        torch.testing.assert_close(cache.ngram, before_ngram, atol=0, rtol=0)
        torch.testing.assert_close(cache.convolution, before_conv, atol=0, rtol=0)

        next_ids = torch.arange(101, 107, device="cuda", dtype=torch.int64)
        proposals = req.req_sampling_params_manager.req_to_next_token_ids
        proposals.fill_(-1)
        expected_mask = []
        for start, count, accepted in zip(starts, counts, accepts + [1]):
            request = requests[start]
            proposals[request, 1:accepted] = next_ids[start : start + accepted - 1]
            expected_mask.extend([1] * accepted + [0] * (count - accepted))
        accept_lengths, accepted_index = verify_mtp_tokens(
            backend,
            next_ids,
            torch.tensor(requests, device="cuda", dtype=torch.int32),
            torch.tensor(starts, device="cuda", dtype=torch.int32),
            torch.tensor(sum((list(range(count)) for count in counts), []), device="cuda", dtype=torch.int32),
        )
        assert accept_lengths.tolist() == accepts + [1]
        assert accepted_index.tolist() == expected_mask
        for start, count, accepted in zip(starts[:2], counts[:2], accepts):
            request = requests[start]
            expected, states = reference_segment(
                residual[start : start + count],
                tokens[start : start + count],
                source,
                config,
                before_ngram[request : request + 1],
                before_conv[request : request + 1],
            )
            torch.testing.assert_close(actual[start : start + count], expected, atol=0.016, rtol=0.016)
            ngram, convolution = states[accepted]
            torch.testing.assert_close(cache.ngram[request], ngram[0], atol=0, rtol=0)
            torch.testing.assert_close(cache.convolution[request], convolution[0], atol=0.016, rtol=0.016)
            assert req.req_to_mtp_state_index[request] == accepted - 1
        torch.testing.assert_close(cache.ngram[7], untouched.ngram, atol=0, rtol=0)
        torch.testing.assert_close(cache.convolution[7], untouched.convolution, atol=0, rtol=0)
    if capture:
        graph.reset()


def test_microbatch_candidates_survive_reverse_verification(config):
    cache, source = make_cache(config)
    speculative = LexicalSpeculativeCache(cache, 3)
    batches = []
    for request, ids in ((5, [19, 63, 23]), (2, [29, 31, 37])):
        meta = metadata([request] * 3, [11, 12, 13], [1] * 3)
        tokens = torch.tensor(ids, device="cuda")
        residual = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
        batch = cache.prepare(meta)
        batch.apply(residual, tokens, cache.layer_weight, 1, None)
        speculative.stage(meta, batch)
        batches.append((request, residual, tokens))
    for request, residual, tokens in reversed(batches):
        _, states = reference_segment(
            residual,
            tokens,
            source,
            config,
            cache.ngram[request : request + 1].clone(),
            cache.convolution[request : request + 1].clone(),
        )
        speculative.commit(
            torch.tensor([request] * 3, device="cuda", dtype=torch.int32),
            torch.tensor([0], device="cuda", dtype=torch.int32),
            torch.tensor([2], device="cuda", dtype=torch.int32),
            7,
        )
        torch.testing.assert_close(cache.ngram[request], states[2][0][0], atol=0, rtol=0)
        torch.testing.assert_close(cache.convolution[request], states[2][1][0], atol=0.016, rtol=0.016)

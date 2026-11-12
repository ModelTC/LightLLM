import pytest
import torch

from lightllm.models.qwen4_exp.lexical_cache import LexicalSnapshot
from .lexical_helpers import make_cache, metadata, reference_segment

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("reset", [False, True])
def test_history_gather_uses_current_request_and_prefix_on_graph_replay(config, reset):
    cache, _ = make_cache(config)
    meta = metadata([1, 3, 7, 7], [1, 10, 2, 2], [1] * 4)
    cache.prepare(meta, reset)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = cache.prepare(meta, reset)
    for request in (2, 4, 5):
        meta.active_rows[:2].fill_(request)
        meta.key_lengths[0] = 1 if request == 4 else 10
        cache.ngram.add_(1)
        cache.convolution.normal_()
        graph.replay()
        rows = meta.active_rows
        empty = rows < 0
        if reset:
            empty |= meta.key_lengths == meta.starts[1:] - meta.starts[:-1]
        ngram = cache.ngram[rows.clamp_min(0).long()].masked_fill(empty[:, None], cache.eos_token_id)
        convolution = cache.convolution[rows.clamp_min(0).long()].masked_fill(empty[:, None, None], 0)
        torch.testing.assert_close(actual.ngram_history, ngram, atol=0, rtol=0)
        torch.testing.assert_close(actual.conv_history, convolution, atol=0, rtol=0)
    graph.reset()


@pytest.mark.parametrize("capture", [False, True])
def test_request_histories_acceptance_padding_and_graphs(config, capture):
    torch.manual_seed(1207)
    cache, source = make_cache(config)

    cache.ngram[5].copy_(torch.tensor([7, 11], device="cuda"))
    cache.convolution[5].normal_()
    prefix = cache.snapshot(5)
    cache.restore(2, prefix)
    requests = [5, 5, 5, 2, 7, 7]
    batch_meta = metadata(requests, [14, 15, 16, 14, 2, 2], [1] * 6)
    accepted = torch.tensor([0, 0, 2, 1, 0, 0], device="cuda", dtype=torch.int32)
    tokens = torch.zeros(6, device="cuda", dtype=torch.int64)
    residual = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16)
    untouched_ngram, untouched_conv = cache.ngram[[0, 7]].clone(), cache.convolution[[0, 7]].clone()

    def run():
        candidate = cache.prepare(batch_meta)
        update = candidate.apply(residual, tokens, cache.layer_weight, 1, None)
        cache.commit(batch_meta, candidate, accepted)
        return update

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
        cache.restore(5, prefix)
        cache.restore(2, prefix)
    ref_ngram = {request: prefix.ngram[None].clone() for request in (5, 2)}
    ref_conv = {request: prefix.convolution[None].clone() for request in (5, 2)}
    for ids in ([19, 63, 23, 29, 31, 37], [41, 43, 47, 63, 5, 7], [11, 13, 17, 19, 23, 29]):
        tokens.copy_(torch.tensor(ids, device="cuda", dtype=torch.int64))
        residual.normal_()
        if capture:
            graph.replay()
            actual = result
        else:
            before_ngram, before_conv = cache.ngram.clone(), cache.convolution.clone()
            candidate = cache.prepare(batch_meta)
            actual = candidate.apply(residual, tokens, cache.layer_weight, 1, None)

            torch.testing.assert_close(cache.ngram, before_ngram, atol=0, rtol=0)
            torch.testing.assert_close(cache.convolution, before_conv, atol=0, rtol=0)
            cache.commit(batch_meta, candidate, accepted)
        for request, start, end, count in ((5, 0, 3, 2), (2, 3, 4, 1)):
            expected, states = reference_segment(
                residual[start:end], tokens[start:end], source, config, ref_ngram[request], ref_conv[request]
            )
            torch.testing.assert_close(actual[start:end], expected, atol=0.016, rtol=0.016)
            ref_ngram[request], ref_conv[request] = states[count]
            torch.testing.assert_close(cache.ngram[request], ref_ngram[request][0], atol=0, rtol=0)
            torch.testing.assert_close(cache.convolution[request], ref_conv[request][0], atol=0.016, rtol=0.016)
        torch.testing.assert_close(cache.ngram[[0, 7]], untouched_ngram, atol=0, rtol=0)
        torch.testing.assert_close(cache.convolution[[0, 7]], untouched_conv, atol=0, rtol=0)
    torch.testing.assert_close(prefix.ngram, torch.tensor([7, 11], device="cuda"), atol=0, rtol=0)


def test_lexical_snapshot_reset_cpu_restore_and_rejection(config):
    torch.manual_seed(1207)
    cache, _ = make_cache(config)
    cache.ngram[3].copy_(torch.tensor([7, 11], device="cuda"))
    cache.convolution[3].normal_()
    snapshot = cache.snapshot(3)
    cache.reset(3)
    assert torch.all(cache.ngram[3] == config["eos_token_id"])
    assert torch.count_nonzero(cache.convolution[3]) == 0
    cache.restore(5, LexicalSnapshot(snapshot.ngram.cpu(), snapshot.convolution.cpu()))
    torch.testing.assert_close(cache.ngram[5], snapshot.ngram)
    torch.testing.assert_close(cache.convolution[5], snapshot.convolution)
    before_ngram, before_conv = cache.ngram.clone(), cache.convolution.clone()
    batch_meta = metadata([5], [12], [3])
    candidate = cache.prepare(batch_meta)
    candidate.apply(
        torch.randn(3, 64, device="cuda", dtype=torch.bfloat16),
        torch.tensor([17, 63, 19], device="cuda"),
        cache.layer_weight,
        1,
        None,
    )
    cache.commit(batch_meta, candidate, torch.zeros(1, device="cuda", dtype=torch.int32))
    torch.testing.assert_close(cache.ngram, before_ngram, atol=0, rtol=0)
    torch.testing.assert_close(cache.convolution, before_conv, atol=0, rtol=0)
    with pytest.raises(ValueError, match="incompatible"):
        cache.restore(3, LexicalSnapshot(snapshot.ngram[:1], snapshot.convolution))
    assert torch.all(cache.ngram[3] == config["eos_token_id"])

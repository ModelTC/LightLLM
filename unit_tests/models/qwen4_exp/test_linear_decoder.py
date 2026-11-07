from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen4_exp.layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight
from lightllm.models.qwen4_exp.layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer
from lightllm.models.qwen4_exp.state import PLEBatch
from lightllm.models.qwen4_exp.batch import PackedBatchMetadata
from lightllm.models.qwen4_exp.lexical_cache import LexicalRequestCache
from lightllm.common.quantization import Quantcfg
from lightllm.common.basemodel.attention.linear.triton import TritonLinearAttBackend
from .helpers import checkpoint
from .reference import reference_step

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("capture", [False, True])
def test_native_ple_gdn_and_moe_cached_steps(config, capture):
    torch.manual_seed(1037)
    source = checkpoint(config, 1)
    weights = Qwen4ExpTransformerLayerWeight(1, torch.bfloat16, config, Quantcfg(config))
    weights.load_hf_weights(dict(source))
    weights.verify_load()
    infer = Qwen4ExpTransformerLayerInfer(1, config)
    channels = 512
    conv = torch.zeros(2, channels, 3, device="cuda", dtype=torch.bfloat16)
    recurrent = torch.zeros(2, 2, 128, 128, device="cuda", dtype=torch.float32)
    state = SimpleNamespace(
        is_prefill=False,
        need_dp_prefill_balance=False,
        dist_group=None,
        batch_size=2,
        b_req_idx=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        att_state_role="main",
        att_state_manager=SimpleNamespace(
            get_state=lambda **kwargs: SimpleNamespace(get_layer_view=lambda **kwargs: (conv, recurrent))
        ),
        req_manager=SimpleNamespace(get_mamba_cache=lambda layer: (conv, recurrent)),
    )

    class Model:
        pass

    model = Model()
    model.config, model.tp_world_size_ = config, 1
    model.is_mtp_draft_model = False
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda is_draft: 0)
    backend = TritonLinearAttBackend(model=model)
    state.decode_att_state1 = backend.create_att_decode_state(state)
    state.decode_att_state1.init_state()
    cache = LexicalRequestCache(weights, 2)
    ref_ngram, ref_ple = cache.ngram.clone(), cache.convolution.clone()
    ref_conv, ref_recurrent = conv.clone(), recurrent.clone()
    ones = torch.ones(2, device="cuda", dtype=torch.int32)
    metadata = PackedBatchMetadata.prepare(state.b_req_idx, ones, ones, 2, 7, 4)

    def run(residual, tokens):
        batch = cache.prepare(metadata)
        state.input_ids, state.qwen4_exp_ple = tokens, {1: batch}
        output = infer.token_forward(residual, state, weights)
        cache.commit(metadata, batch)
        return output

    if capture:
        residual = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
        tokens = torch.zeros(2, device="cuda", dtype=torch.int64)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run(residual, tokens)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = run(residual, tokens)
        conv.zero_()
        recurrent.zero_()
        for request in range(2):
            cache.reset(request)
    for ids in [[7, 11], [63, 15], [19, 23]]:
        if capture:
            residual.normal_()
            tokens.copy_(torch.tensor(ids, device="cuda", dtype=torch.int64))
            graph.replay()
            actual = captured_output
        else:
            residual = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
            tokens = torch.tensor(ids, device="cuda", dtype=torch.int64)
            actual = run(residual, tokens)
        expected, ref_ngram, ref_ple, ref_conv, ref_recurrent = reference_step(
            residual, tokens, source, config, ref_ngram, ref_ple, ref_conv, ref_recurrent
        )
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.016)
        torch.testing.assert_close(cache.ngram, ref_ngram, atol=0, rtol=0)
        torch.testing.assert_close(cache.convolution, ref_ple, atol=0.016, rtol=0.016)
        torch.testing.assert_close(conv, ref_conv, atol=0.016, rtol=0.016)


def test_native_packed_gdn_ple_prefill_matches_recurrent_reference(config):
    from lightllm.models.qwen3next.model import Qwen3NextTpPartModel

    torch.manual_seed(1037)

    Qwen3NextTpPartModel._init_triton(None)
    source = checkpoint(config, 1)
    weights = Qwen4ExpTransformerLayerWeight(1, torch.bfloat16, config, Quantcfg(config))
    weights.load_hf_weights(dict(source))
    weights.verify_load()
    infer = Qwen4ExpTransformerLayerInfer(1, config)
    conv = torch.zeros(2, 512, 3, device="cuda", dtype=torch.bfloat16)
    recurrent = torch.zeros(2, 2, 128, 128, device="cuda", dtype=torch.float32)
    starts = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    tokens = torch.tensor([7, 63, 19, 11, 23], device="cuda", dtype=torch.int64)
    residual = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    batch = PLEBatch(starts, weights.ngram_hasher.component.empty_history(2), weights.ple.component.empty_history(2))
    state = SimpleNamespace(
        is_prefill=True,
        need_dp_prefill_balance=False,
        dist_group=None,
        batch_size=2,
        b_req_idx=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        b1_cu_q_seq_len=starts,
        b_ready_cache_len=torch.zeros(2, device="cuda", dtype=torch.int32),
        att_state_role="main",
        att_state_manager=SimpleNamespace(
            get_state=lambda **kwargs: SimpleNamespace(get_layer_view=lambda **kwargs: (conv, recurrent))
        ),
        req_manager=SimpleNamespace(get_mamba_cache=lambda layer: (conv, recurrent)),
        input_ids=tokens,
        qwen4_exp_ple={1: batch},
    )

    class Model:
        pass

    model = Model()
    model.config, model.tp_world_size_ = config, 1
    model.is_mtp_draft_model = False
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda is_draft: 0)
    backend = TritonLinearAttBackend(model=model)
    state.prefill_att_state1 = backend.create_att_prefill_state(state)
    state.prefill_att_state1.init_state()
    actual = infer.context_forward(residual, state, weights)
    next_ngram, next_ple = batch.histories_after(weights)
    expected = []
    for request, (start, end) in enumerate([(0, 3), (3, 5)]):
        ng = weights.ngram_hasher.component.empty_history(1)
        ple = weights.ple.component.empty_history(1)
        cv = torch.zeros(1, 512, 3, device="cuda", dtype=torch.bfloat16)
        rc = torch.zeros(1, 2, 128, 128, device="cuda", dtype=torch.float32)
        for token in range(start, end):
            out, ng, ple, cv, rc = reference_step(
                residual[token : token + 1], tokens[token : token + 1], source, config, ng, ple, cv, rc
            )
            expected.append(out)
        torch.testing.assert_close(next_ngram[request : request + 1], ng, atol=0, rtol=0)
        torch.testing.assert_close(next_ple[request : request + 1], ple, atol=0.016, rtol=0.016)
    torch.testing.assert_close(actual, torch.cat(expected), atol=0.016, rtol=0.016)


def test_native_ple_empty_packed_batch_preserves_histories(config):
    source = checkpoint(config, 1)
    weights = Qwen4ExpTransformerLayerWeight(1, torch.bfloat16, config, Quantcfg(config))
    weights.load_hf_weights(dict(source))
    weights.verify_load()
    starts = torch.zeros(3, device="cuda", dtype=torch.int32)
    ngram = weights.ngram_hasher.component.empty_history(2)
    conv = weights.ple.component.empty_history(2)
    batch = PLEBatch(starts, ngram, conv)
    update = batch.apply(
        torch.empty(0, 64, device="cuda", dtype=torch.bfloat16),
        torch.empty(0, device="cuda", dtype=torch.int64),
        weights,
        1,
        None,
    )
    assert update.shape == (0, 64)
    next_ngram, next_conv = batch.histories_after(weights)
    torch.testing.assert_close(next_ngram, ngram, atol=0, rtol=0)
    torch.testing.assert_close(next_conv, conv, atol=0, rtol=0)

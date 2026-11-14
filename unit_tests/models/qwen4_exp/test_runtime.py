import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.models.qwen4_exp.layer_weights.transformer_layer_weight import Qwen4ExpTransformerLayerWeight
from lightllm.models.qwen4_exp.model import Qwen4ExpTpPartModel
from lightllm.models.qwen4_exp.layer_infer.transformer_layer_infer import Qwen4ExpTransformerLayerInfer
from lightllm.common.quantization import Quantcfg
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from .helpers import checkpoint
from .reference import reference_step, reference_layer, reference_model
from .runtime import state_for, assemble_model
from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_decode_graph_preparation_reads_current_positions_and_lexical_history(runtime):
    old = state_for(runtime, [1, 3], [3, 2], [7, 11], graph_mode="capture")
    old.prepare_qwen_state()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        old.prepare_qwen_state()
    cache = runtime.req.lexical_caches[1]
    for requests, delta in (([3, 1], [-1, 7]), ([1, 3], [4, 0])):
        new = state_for(runtime, requests, [4, 4], [13, 17], graph_mode="replay")
        new.b_position_delta.copy_(torch.tensor(delta, device="cuda"))
        cache.ngram.random_(0, 63)
        cache.convolution.normal_()
        before = cache.snapshot(1)
        old.copy_for_cuda_graph(new)
        graph.replay()
        positions = torch.tensor(delta, device="cuda") + 3
        torch.testing.assert_close(old.position_ids, positions.expand(3, -1).to(torch.int32), atol=0, rtol=0)
        torch.testing.assert_close(old.qwen4_exp_batch.cos, runtime.model._cos_cached[positions].repeat(1, 2))
        torch.testing.assert_close(old.qwen4_exp_batch.sin, runtime.model._sin_cached[positions].repeat(1, 2))
        batch = old.qwen4_exp_ple[1]
        torch.testing.assert_close(batch.ngram_history, cache.ngram[requests], atol=0, rtol=0)
        torch.testing.assert_close(batch.conv_history, cache.convolution[requests], atol=0, rtol=0)
        torch.testing.assert_close(cache.ngram[1], before.ngram, atol=0, rtol=0)
        torch.testing.assert_close(cache.convolution[1], before.convolution, atol=0, rtol=0)


@pytest.mark.parametrize("runtime", [{}, {"ssm_state_mode": "replay"}], ids=["native", "replay"], indirect=True)
def test_inference_state_prefill_decode_and_automatic_commit(runtime, config):
    Qwen3NextTpPartModel._init_triton(None)
    req, weight = runtime.req, runtime.weight
    for request in (1, 3):
        req.init_linear_att_state(SimpleNamespace(req_idx=request))
    state = state_for(runtime, [1, 3], [3, 2], [7, 63, 19, 11, 23], ready=[0, 0])
    assert state.qwen4_exp_metadata.starts.tolist() == [0, 3, 5, 5, 5, 5]
    assert state.qwen4_exp_batch.slot_table.data_ptr() == req.req_to_token_indexs.data_ptr()
    torch.testing.assert_close(
        runtime.mem.qsa_key_cos[state.mem_index.long()], state.qwen4_exp_batch.cos, atol=0, rtol=0
    )
    infer = Qwen4ExpTransformerLayerInfer(1, config)
    residual = torch.randn(5, 64, device="cuda", dtype=torch.bfloat16)
    reference = {}
    expected = []
    for request, start, end in ((1, 0, 3), (3, 3, 5)):
        ngram = weight.ngram_hasher.component.empty_history(1)
        ple = weight.ple.component.empty_history(1)
        conv = torch.zeros(1, 512, 3, device="cuda", dtype=torch.bfloat16)
        ssm = torch.zeros(1, 2, 128, 128, device="cuda", dtype=torch.float32)
        for row in range(start, end):
            output, ngram, ple, conv, ssm = reference_step(
                residual[row : row + 1], state.input_ids[row : row + 1], runtime.source, config, ngram, ple, conv, ssm
            )
            expected.append(output)
        reference[request] = (ngram, ple, conv, ssm)
    actual = infer.context_forward(residual, state, weight)
    torch.testing.assert_close(actual, torch.cat(expected), atol=0.016, rtol=0.016)

    state = state_for(runtime, [1, 3], [4, 3], [29, 31])
    residual = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    actual = infer.token_forward(residual, state, weight)
    for row, request in enumerate((1, 3)):
        expected, ngram, ple, _, _ = reference_step(
            residual[row : row + 1], state.input_ids[row : row + 1], runtime.source, config, *reference[request]
        )
        torch.testing.assert_close(actual[row : row + 1], expected, atol=0.016, rtol=0.016)
        torch.testing.assert_close(req.lexical_caches[1].ngram[request], ngram[0], atol=0, rtol=0)
        torch.testing.assert_close(req.lexical_caches[1].convolution[request], ple[0], atol=0.016, rtol=0.016)


def test_native_prefill_graph_state_copy_changes_request_count(runtime, config):
    source = checkpoint(config, 3)
    weight = Qwen4ExpTransformerLayerWeight(3, torch.bfloat16, config, Quantcfg(config))
    weight.load_hf_weights(dict(source))
    weight.verify_load()
    infer = Qwen4ExpTransformerLayerInfer(3, config)
    old = state_for(runtime, [1], [4], [7, 11, 13, 17], ready=[0])
    residual = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        infer.context_forward(residual, old, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = infer.context_forward(residual, old, weight)
    pointers = [
        getattr(old.qwen4_exp_metadata, name).data_ptr() for name in old.qwen4_exp_metadata.__dataclass_fields__
    ]
    before = runtime.mem.get_index_kv_buffer(old.mem_index)
    new = state_for(runtime, [2, 3], [2, 2], [19, 23, 29, 31], ready=[0, 0])
    old.copy_for_prefill_cuda_graph(new)
    residual.normal_()
    graph.replay()
    expected = reference_layer(residual, source, config, new.qwen4_exp_batch.cos, new.qwen4_exp_batch.sin, [2, 2])
    torch.testing.assert_close(output, expected, atol=0.016, rtol=0.016)
    assert pointers == [
        getattr(old.qwen4_exp_metadata, name).data_ptr() for name in old.qwen4_exp_metadata.__dataclass_fields__
    ]
    for name, values in runtime.mem.get_index_kv_buffer(torch.arange(4, device="cuda")).items():
        torch.testing.assert_close(values, before[name], atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize(
    "runtime",
    [
        {"page_size": 4, "ssm_state_mode": "replay"},
        {"page_size": 4, "llm_prefill_att_backend": ["triton", "flashinfer"]},
        {"page_size": 4, "ssm_state_mode": "replay", "llm_prefill_att_backend": ["triton", "flashinfer"]},
    ],
    indirect=True,
)
def test_prefill_graph_reuses_current_attention_state(runtime, config):
    from lightllm.common.basemodel.prefill_cuda_graph import PrefillCudaGraph

    Qwen3NextTpPartModel._init_triton(None)
    model, source = assemble_model(runtime, config)
    for request in (1, 2, 3):
        runtime.req.init_linear_att_state(SimpleNamespace(req_idx=request))

    def run(state):
        state.hidden_collector = NoopHiddenCollector()
        return model._context_forward(state).logits.clone()

    state = state_for(runtime, [1], [4], [7, 11, 13, 17], ready=[0])
    expected = run(state)
    runtime.req.init_linear_att_state(SimpleNamespace(req_idx=1))
    model.prefill_graph = PrefillCudaGraph(None, 1)
    actual = run(state)
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)

    state = state_for(runtime, [2, 3], [2, 2], [19, 23, 29, 31], ready=[0, 0])
    actual = run(state)
    expected = reference_model(
        state.input_ids, state.qwen4_exp_batch.cos, state.qwen4_exp_batch.sin, [2, 2], source, config
    )
    torch.testing.assert_close(actual, expected[[1, 3]], atol=0.025, rtol=0.025)
    prefix = state
    state = state_for(runtime, [3], [6], [37, 41, 43, 47], ready=[2])
    state.max_kv_seq_len = 6
    actual = run(state)
    expected = reference_model(
        torch.tensor([29, 31, 37, 41, 43, 47], device="cuda"),
        torch.cat((prefix.qwen4_exp_batch.cos[2:], state.qwen4_exp_batch.cos)),
        torch.cat((prefix.qwen4_exp_batch.sin[2:], state.qwen4_exp_batch.sin)),
        [6],
        source,
        config,
    )
    torch.testing.assert_close(actual, expected[[-1]], atol=0.025, rtol=0.025)


@pytest.mark.parametrize(
    "runtime",
    [
        {"page_size": 4, "llm_prefill_att_backend": ["triton", "flashinfer"]},
        {
            "page_size": 4,
            "ssm_state_mode": "replay",
            "mtp_step": 3,
            "llm_prefill_att_backend": ["triton", "flashinfer"],
        },
    ],
    indirect=True,
)
def test_flashinfer_prefill_graph_replays_and_falls_back_for_changed_batch(runtime, config, monkeypatch):
    from lightllm.common.basemodel.prefill_cuda_graph import PrefillCudaGraph

    Qwen3NextTpPartModel._init_triton(None)
    model, _ = assemble_model(runtime, config)
    requests = (1, 2, 3)
    cases = [
        ([1], [4], [7, 11, 13, 17], [0]),
        ([1], [8], [19, 23, 29, 31], [4]),
        ([3, 2], [2, 2], [37, 41, 43, 47], [0, 0]),
        ([2], [6], [11, 13, 17, 19], [2]),
    ]
    bounded = model.att_state_manager.get_bounded_attention_states(model.att_state_role)
    linear = next(state for state in bounded if state.get_state_type().startswith("gdn_"))

    def run():
        results = []
        for ids, lengths, tokens, ready in cases:
            state = state_for(runtime, ids, lengths, tokens, ready=ready)
            state.hidden_collector = NoopHiddenCollector()
            logits = model._context_forward(state).logits.clone()
            buffers = [
                tensor[list(requests)].clone()
                for layer in linear.config.logical_layer_ids
                for tensor in linear.get_layer_view(logical_layer_id=layer)
            ]
            results.append((logits, buffers))
        return results

    for request in requests:
        runtime.req.init_linear_att_state(SimpleNamespace(req_idx=request))
    expected = run()
    for request in requests:
        runtime.req.init_linear_att_state(SimpleNamespace(req_idx=request))

    calls = []
    kernel = model.prefill_att_backend1.prefill_kernel

    def tracked(**kwargs):
        calls.append(torch.cuda.is_current_stream_capturing())
        return kernel(**kwargs)

    monkeypatch.setattr(model.prefill_att_backend1, "prefill_kernel", tracked)
    model.prefill_graph = PrefillCudaGraph(None, 1)
    actual = run()
    linear_layers = len(linear.config.logical_layer_ids)
    assert calls.count(True) == linear_layers
    assert calls.count(False) == linear_layers
    for (logits, buffers), (expected_logits, expected_buffers) in zip(actual, expected):
        torch.testing.assert_close(logits, expected_logits, atol=0.025, rtol=0.025)
        for value, reference in zip(buffers, expected_buffers):
            torch.testing.assert_close(value, reference, atol=0.003, rtol=0.03)


def test_native_config_normalization_and_periodic_layout(runtime, config, tmp_path):
    outer = dict(text_config=dict(config), vision_config={"hidden_size": 16})
    outer["text_config"]["rope_parameters"] = dict(
        rope_type="default", rope_theta=10000, partial_rotary_factor=0.5, mrope_section=[1, 1, 0]
    )
    (tmp_path / "config.json").write_text(json.dumps(outer))
    model = Qwen4ExpTpPartModel.__new__(Qwen4ExpTpPartModel)
    model.weight_dir_, model.finetune_config, model.tp_world_size_ = str(tmp_path), None, 1
    model._init_config()
    assert model.config["n_layer"] == 4 and model.config["n_embed"] == 32
    assert model.config["rope_theta"] == 10000
    assert model.num_kv_heads == 2
    invalid = dict(config, layer_types=["full_attention"] * 4)
    with pytest.raises(ValueError, match="periodic"):
        Qwen4ExpTpPartModel.normalize_model_config(invalid)


def test_native_decode_graph_replay_updates_positions_slots_and_request_order(runtime, config):
    source = checkpoint(config, 3)
    weight = Qwen4ExpTransformerLayerWeight(3, torch.bfloat16, config, Quantcfg(config))
    weight.load_hf_weights(dict(source))
    weight.verify_load()
    infer = Qwen4ExpTransformerLayerInfer(3, config)
    prefix = state_for(runtime, [1, 3], [4, 4], [7, 11, 13, 17, 19, 23, 29, 31], ready=[0, 0])
    prefix_residual = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    infer.context_forward(prefix_residual, prefix, weight)
    old = state_for(runtime, [1, 3], [5, 5], [37, 41])
    residual = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        infer.token_forward(residual, old, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = infer.token_forward(residual, old, weight)
    previous_residual = residual.clone()
    previous_cos, previous_sin = old.qwen4_exp_batch.cos.clone(), old.qwen4_exp_batch.sin.clone()
    new = state_for(runtime, [3, 1], [6, 6], [43, 47])
    old.copy_for_cuda_graph(new)
    residual.normal_()
    graph.replay()
    all_residual, all_cos, all_sin = [], [], []
    for row, (start, end) in enumerate(((0, 4), (4, 8))):
        incoming = 1 - row
        all_residual.extend(
            [prefix_residual[start:end], previous_residual[row : row + 1], residual[incoming : incoming + 1]]
        )
        all_cos.extend(
            [
                prefix.qwen4_exp_batch.cos[start:end],
                previous_cos[row : row + 1],
                new.qwen4_exp_batch.cos[incoming : incoming + 1],
            ]
        )
        all_sin.extend(
            [
                prefix.qwen4_exp_batch.sin[start:end],
                previous_sin[row : row + 1],
                new.qwen4_exp_batch.sin[incoming : incoming + 1],
            ]
        )
    expected = reference_layer(torch.cat(all_residual), source, config, torch.cat(all_cos), torch.cat(all_sin), [6, 6])
    torch.testing.assert_close(output, expected[[11, 5]], atol=0.016, rtol=0.016)


@pytest.mark.parametrize(
    "runtime",
    [
        {},
        {"ssm_state_mode": "replay"},
        {"page_size": 4},
        {"page_size": 4, "ssm_state_mode": "replay"},
        {"page_size": 4, "llm_prefill_att_backend": ["triton", "flashinfer"]},
        {"page_size": 4, "ssm_state_mode": "replay", "llm_prefill_att_backend": ["triton", "flashinfer"]},
    ],
    ids=["native", "replay", "compact-native", "compact-replay", "flashinfer-native", "flashinfer-replay"],
    indirect=True,
)
@pytest.mark.parametrize("graph_preparation", [False, True])
def test_complete_small_model_prefill_and_cached_logits(runtime, config, graph_preparation):
    Qwen3NextTpPartModel._init_triton(None)
    model, source = assemble_model(runtime, config)
    for request in (1, 3):
        runtime.req.init_linear_att_state(SimpleNamespace(req_idx=request))
    state = state_for(runtime, [1, 3], [3, 2], [7, 63, 19, 11, 23], ready=[0, 0])
    state.hidden_collector = NoopHiddenCollector()
    output = model._context_forward(state)
    expected = reference_model(
        state.input_ids, state.qwen4_exp_batch.cos, state.qwen4_exp_batch.sin, [3, 2], source, config
    )
    torch.testing.assert_close(output.logits, expected[[2, 4]], atol=0.025, rtol=0.025)
    decode = state_for(runtime, [1, 3], [4, 3], [29, 31])
    decode.hidden_collector = NoopHiddenCollector()
    output = model._token_forward(decode)
    all_tokens = torch.tensor([7, 63, 19, 29, 11, 23, 31], device="cuda", dtype=torch.int64)
    all_cos = torch.cat(
        [
            state.qwen4_exp_batch.cos[:3],
            decode.qwen4_exp_batch.cos[:1],
            state.qwen4_exp_batch.cos[3:],
            decode.qwen4_exp_batch.cos[1:],
        ]
    )
    all_sin = torch.cat(
        [
            state.qwen4_exp_batch.sin[:3],
            decode.qwen4_exp_batch.sin[:1],
            state.qwen4_exp_batch.sin[3:],
            decode.qwen4_exp_batch.sin[1:],
        ]
    )
    expected = reference_model(all_tokens, all_cos, all_sin, [4, 3], source, config)
    torch.testing.assert_close(output.logits, expected[[3, 6]], atol=0.025, rtol=0.025)

    third = state_for(runtime, [1, 3], [5, 4], [37, 41], graph_mode="capture" if graph_preparation else None)
    third.hidden_collector = NoopHiddenCollector()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = model._token_forward(third)
    graph.replay()
    third_cos, third_sin = third.qwen4_exp_batch.cos.clone(), third.qwen4_exp_batch.sin.clone()
    all_tokens = torch.tensor([7, 63, 19, 29, 37, 11, 23, 31, 41], device="cuda", dtype=torch.int64)
    all_cos = torch.cat([all_cos[:4], third_cos[:1], all_cos[4:], third_cos[1:]])
    all_sin = torch.cat([all_sin[:4], third_sin[:1], all_sin[4:], third_sin[1:]])
    expected = reference_model(all_tokens, all_cos, all_sin, [5, 4], source, config)
    torch.testing.assert_close(captured.logits, expected[[4, 8]], atol=0.025, rtol=0.025)
    fourth = state_for(runtime, [3, 1], [5, 6], [47, 43], graph_mode="replay" if graph_preparation else None)
    fourth.hidden_collector = NoopHiddenCollector()
    third.copy_for_cuda_graph(fourth)
    graph.replay()
    all_tokens = torch.tensor([7, 63, 19, 29, 37, 43, 11, 23, 31, 41, 47], device="cuda", dtype=torch.int64)
    current = third if graph_preparation else fourth
    all_cos = torch.cat([all_cos[:5], current.qwen4_exp_batch.cos[1:], all_cos[5:], current.qwen4_exp_batch.cos[:1]])
    all_sin = torch.cat([all_sin[:5], current.qwen4_exp_batch.sin[1:], all_sin[5:], current.qwen4_exp_batch.sin[:1]])
    expected = reference_model(all_tokens, all_cos, all_sin, [6, 5], source, config)
    torch.testing.assert_close(captured.logits, expected[[10, 5]], atol=0.025, rtol=0.025)


def test_fresh_prefill_ignores_stale_lexical_slot_without_mutating_accepted_state(runtime):
    cache = runtime.req.lexical_caches[1]
    cache.ngram[1].fill_(17)
    cache.convolution[1].normal_()
    accepted = cache.snapshot(1)
    fresh = state_for(runtime, [1], [2], [7, 11], ready=[0])
    batch = fresh.qwen4_exp_ple[1]
    assert torch.all(batch.ngram_history == 63)
    assert torch.count_nonzero(batch.conv_history) == 0
    torch.testing.assert_close(cache.ngram[1], accepted.ngram, atol=0, rtol=0)
    torch.testing.assert_close(cache.convolution[1], accepted.convolution, atol=0, rtol=0)
    resumed = state_for(runtime, [1], [4], [13, 19], ready=[2])
    batch = resumed.qwen4_exp_ple[1]
    torch.testing.assert_close(batch.ngram_history[0], accepted.ngram, atol=0, rtol=0)
    torch.testing.assert_close(batch.conv_history[0], accepted.convolution, atol=0, rtol=0)

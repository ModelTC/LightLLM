from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer


@pytest.mark.parametrize(
    "balance,ep,tpsp,tp,redundancy,error",
    [
        (True, False, False, 2, 0, "enable_dp_prefill_balance"),
        (False, True, True, 2, 0, "FFN and mHC token layouts"),
        (False, True, False, 2, 2, "redundant experts"),
        (False, True, True, 1, 0, None),
        (False, True, False, 2, 0, None),
        (False, False, True, 2, 0, None),
    ],
)
def test_unsupported_parallel_layouts_fail_before_inference(monkeypatch, balance, ep, tpsp, tp, redundancy, error):
    monkeypatch.setattr("lightllm.models.deepseek_v4.model.get_redundancy_expert_num", lambda: redundancy)
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.load_way, model.tp_world_size_ = "HF", tp
    model.config = {"num_attention_heads": 64, "o_groups": 8, "index_n_heads": 64}
    model.args = SimpleNamespace(enable_dp_prefill_balance=balance, enable_ep_moe=ep, enable_tpsp_mix_mode=tpsp)
    if error:
        with pytest.raises(ValueError, match=error):
            model._verify_params()
    else:
        model._verify_params()


def test_memory_check_rejects_sequence_larger_than_full_token_pool():
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.mem_manager = SimpleNamespace(size=60416)
    model.batch_max_tokens = 8192
    model.max_seq_length = 262169
    model.args = SimpleNamespace(performance_mode=None)

    with pytest.raises(AssertionError, match="max_total_token_num must be >= max_seq_length"):
        model._check_mem_size()


def test_memory_check_still_requires_one_prefill_batch():
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.mem_manager = SimpleNamespace(size=8192)
    model.batch_max_tokens = 8192
    model.max_seq_length = 262169
    model.args = SimpleNamespace(performance_mode=None)

    with pytest.raises(AssertionError, match="greater than batch_max_tokens"):
        model._check_mem_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("deferred", [False, True])
def test_draft_hidden_collector_materializes_mhc_streams(monkeypatch, deferred):
    from lightllm.common.basemodel.hidden_collector import LayerHiddenCollector
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.models.deepseek_v4.model import DeepseekV4LayerHiddenCollector

    monkeypatch.setattr(LayerHiddenCollector, "_load_layer_ids", lambda self: frozenset([0]))
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.config = {"hc_mult": 4, "hidden_size": 8}
    model.layers_num = 2
    model.pre_infer = SimpleNamespace(_tpsp_allgather=lambda input, infer_state: input)
    model.mtp_manager = MtpManager.__new__(MtpManager)
    model.mtp_manager.args = SimpleNamespace(mtp_mode="dspark")
    model._init_hidden_collector()
    assert isinstance(model.hidden_collector_prototype, DeepseekV4LayerHiddenCollector)
    collector = model.hidden_collector_prototype.new_instance()

    torch.manual_seed(6103)
    residual = torch.randn((2, 4, 8), dtype=torch.bfloat16, device="cuda")
    output = torch.randn((2, 8), dtype=torch.bfloat16, device="cuda")
    post_mix = torch.randn((2, 4), device="cuda")
    residual_mix = torch.randn((2, 4, 4), device="cuda")
    streams = (
        torch.einsum("bsh,bst->bth", residual.float(), residual_mix) + output.float()[:, None] * post_mix[..., None]
    )
    streams = streams.bfloat16()
    hidden = (output, residual.flatten(1), residual_mix, post_mix) if deferred else streams.flatten(1)

    collector.add(layer_index=1, hidden=None)
    collector.add(layer_index=0, hidden=hidden)
    actual = collector.finish_output(SimpleNamespace(need_dp_prefill_balance=False)).spec_hidden
    torch.testing.assert_close(actual, streams.mean(dim=1), rtol=0.02, atol=0.02)
    assert not model.hidden_collector_prototype.layer_hiddens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("prefill", [False, True], ids=["decode", "prefill"])
@pytest.mark.parametrize("truncated", [False, True], ids=["full", "truncated"])
def test_mhc_layers_and_head_cuda_graph_replay(monkeypatch, tokens, prefill, truncated):
    import triton

    from lightllm.models.deepseek_v4.layer_infer.post_layer_infer import DeepseekV4PostLayerInfer
    from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
    from unit_tests.common.basemodel.triton_kernel.mhc.test_mhc import _hc_post_reference, _hc_pre_reference

    torch.manual_seed(6104)
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    streams, hidden = 4, 4096
    cfg = {"hc_mult": streams, "hidden_size": hidden, "rms_norm_eps": 1e-5, "hc_eps": 1e-6}
    residual = torch.randn(tokens, streams * hidden, device="cuda", dtype=torch.bfloat16)
    layers = []
    weights = []
    for index in range(2):
        layer = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
        layer.hc_mult, layer.embed_dim_ = streams, hidden
        layer.eps_, layer.hc_eps, layer.sinkhorn_iters = cfg["rms_norm_eps"], cfg["hc_eps"], 20
        layer.is_last_layer = index == 1 and not truncated
        layer.context_attention_forward = layer.token_attention_forward = lambda x, *_: x * 0.125
        layer._ffn = lambda x, *_: x * 0.25
        layers.append(layer)
        weight = SimpleNamespace()
        for prefix in ("attn", "ffn"):
            setattr(
                weight,
                f"hc_{prefix}_fn_",
                SimpleNamespace(weight=torch.randn(24, streams * hidden, device="cuda") * 0.005),
            )
            setattr(weight, f"hc_{prefix}_scale_", SimpleNamespace(weight=torch.randn(3, device="cuda")))
            setattr(weight, f"hc_{prefix}_base_", SimpleNamespace(weight=torch.randn(24, device="cuda")))
            setattr(
                weight,
                f"{prefix}_norm_",
                SimpleNamespace(weight=torch.randn(hidden, device="cuda", dtype=torch.bfloat16)),
            )
        weights.append(weight)

    post = DeepseekV4PostLayerInfer.__new__(DeepseekV4PostLayerInfer)
    post.alloc_tensor = torch.empty
    post_weight = SimpleNamespace(
        network_config_=cfg,
        hc_head_fn_=SimpleNamespace(weight=torch.randn(streams, streams * hidden, device="cuda") * 0.005),
        hc_head_scale_=SimpleNamespace(weight=torch.randn(1, device="cuda")),
        hc_head_base_=SimpleNamespace(weight=torch.randn(streams, device="cuda")),
    )
    # Keep real mHC calls and replace the logits path with its input.
    monkeypatch.setattr(LlamaPostLayerInfer, "token_forward", lambda self, x, *_: x)

    def run():
        state = residual
        for layer, weight in zip(layers, weights):
            forward = layer.context_forward if prefill else layer.token_forward
            state = forward(state, None, weight)
            if isinstance(state, tuple):
                x, current_residual, residual_mix, post_mix = state
                assert x.shape == (tokens, hidden)
                assert current_residual.shape == (tokens, streams * hidden)
                assert residual_mix.shape == (tokens, streams, streams)
                assert post_mix.shape == (tokens, streams)
        assert isinstance(state, tuple) == truncated
        return post.token_forward(state, None, post_weight)

    def reference():
        state = residual
        for weight in weights:
            for prefix, factor in (("attn", 0.125), ("ffn", 0.25)):
                x, mix, post_mix = _hc_pre_reference(
                    state,
                    getattr(weight, f"hc_{prefix}_fn_").weight,
                    getattr(weight, f"hc_{prefix}_scale_").weight,
                    getattr(weight, f"hc_{prefix}_base_").weight,
                    streams,
                    cfg["rms_norm_eps"],
                    cfg["hc_eps"],
                    20,
                )
                normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + cfg["rms_norm_eps"])
                normalized = (normalized * getattr(weight, f"{prefix}_norm_").weight.float()).bfloat16()
                state = _hc_post_reference(normalized * factor, state, mix, post_mix, streams)
        inv_rms = torch.rsqrt(state.float().square().mean(-1, keepdim=True) + cfg["rms_norm_eps"])
        gates = (
            torch.sigmoid(
                (state.float() @ post_weight.hc_head_fn_.weight.T) * inv_rms * post_weight.hc_head_scale_.weight
                + post_weight.hc_head_base_.weight
            )
            + cfg["hc_eps"]
        )
        return (state.view(tokens, streams, hidden).float() * gates[..., None]).sum(1).bfloat16()

    torch.testing.assert_close(run(), reference(), atol=0.06, rtol=0.04)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(2):
        residual.normal_()
        graph.replay()
        torch.testing.assert_close(actual, reference(), atol=0.06, rtol=0.04)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads,groups,fp8", [(16, 1, False), (32, 2, False), (128, 8, False), (16, 1, True)])
def test_output_projection_accepts_attention_view(heads, groups, fp8):
    torch.manual_seed(6102)
    tokens = 5
    padded_heads = (heads + 63) // 64 * 64
    padded = torch.randn((tokens, padded_heads, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    attention_view = padded[:, :heads]
    compact = attention_view.contiguous().clone()
    angles = torch.randn((tokens, 32), device="cuda")
    cos, sin = angles.cos().bfloat16(), angles.sin().bfloat16()
    weight_a = torch.randn((groups, heads * 512 // groups, 8), dtype=padded.dtype, device=padded.device) * 0.1
    weight_b = torch.randn((groups * 8, 16), dtype=padded.dtype, device=padded.device) * 0.1
    bmm_inputs = []

    def bmm(input):
        bmm_inputs.append(input)
        return torch.bmm(input, weight_a)

    def mm(input):
        # The FP8 activation quantizer requires a contiguous input.
        assert input.is_contiguous()
        return input @ weight_a[0]

    layer_weight = SimpleNamespace(
        o_proj_fp8=fp8,
        wo_a_=SimpleNamespace(bmm=bmm, mm=mm),
        wo_b_=SimpleNamespace(mm=lambda input: input @ weight_b),
    )
    layer = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
    layer.qk_rope_head_dim = 64
    layer.tp_groups = groups
    layer._select_rope = lambda infer_state: (cos, sin)
    layer._tpsp_reduce = lambda input, infer_state: input

    expected = layer._get_o(compact, None, layer_weight)
    actual = layer._get_o(attention_view, None, layer_weight)

    torch.testing.assert_close(actual, expected)
    if not fp8:
        assert bmm_inputs[-1].untyped_storage().data_ptr() == padded.untyped_storage().data_ptr()


def test_runtime_and_storage_share_normalized_layout(monkeypatch):
    from lightllm.models.deepseek_v4 import model as implementation

    events = []
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.config = {
        "n_layer": 3,
        "compress_ratios": [4, 128, 0],
        "head_dim": 512,
        "index_head_dim": 128,
        "sliding_window": 256,
        "qk_rope_head_dim": 64,
    }
    model.args = SimpleNamespace(
        cpu_cache_token_page_size=2048,
        run_mode="normal",
        mtp_step=3,
        enable_prefill_microbatch_overlap=True,
        llm_kv_type="None",
    )
    model.max_req_num, model.batch_max_tokens, model.tp_world_size_ = 2, 2048, 1
    model.max_total_token_num, model.mem_fraction, model.data_type = 8192, 0.9, torch.bfloat16
    monkeypatch.setattr("lightllm.utils.envs_utils.get_added_mtp_kv_layer_num", lambda: 0)
    monkeypatch.setattr(implementation, "get_env_start_args", lambda: model.args)

    def state(count, config, **kwargs):
        events.append("runtime")
        return SimpleNamespace(cache_config=config)

    def memory(size, *, cache_config, **kwargs):
        events.append("storage")
        return SimpleNamespace(cache_config=cache_config)

    monkeypatch.setattr(implementation, "DeepseekV4StateManager", state)
    monkeypatch.setattr(implementation, "DeepseekV4MemoryManager", memory)
    model._init_cache_coordinator()
    model._init_mem_manager()
    assert events == ["runtime", "storage"]
    assert model.cache_coordinator.state_manager.cache_config is model.cache_config
    assert model.mem_manager.cache_config is model.cache_config
    assert model.cache_config.sliding_window == 256
    model.args.llm_kv_type = "int8kv"
    with pytest.raises(ValueError, match="packed FP8"):
        model.get_mem_manager_class()

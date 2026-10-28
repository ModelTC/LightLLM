from types import SimpleNamespace
import ast
import inspect
import textwrap

import pytest
import torch

from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.models.deepseek_v4.layer_infer.transformer_layer_infer import DeepseekV4TransformerLayerInfer
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight


@pytest.mark.parametrize("method_name", ["_ffn", "overlap_tpsp_context_forward", "overlap_tpsp_token_forward"])
def test_moe_calls_match_shared_weight_interface(method_name):
    method = getattr(DeepseekV4TransformerLayerInfer, method_name)
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("experts_with_topk", "prefilled_group_gemm", "masked_group_gemm")
    ]
    assert calls
    for call in calls:
        signature = inspect.signature(getattr(FusedMoeWeight, call.func.attr))
        signature.bind(None, *([None] * len(call.args)), **{keyword.arg: None for keyword in call.keywords})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rotary_cache_keeps_unscaled_swa_and_yarn_compression():
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.config = dict(
        qk_rope_head_dim=64,
        rope_theta=10000,
        compress_rope_theta=160000,
        max_position_embeddings=8193,
        rope_scaling=dict(type="yarn", factor=16, original_max_position_embeddings=65536, beta_fast=32, beta_slow=1),
    )
    model.max_seq_length = 8193
    model.layers_infer = [SimpleNamespace(compress_ratio=ratio, index_infer=SimpleNamespace()) for ratio in (0, 4, 128)]
    model._init_to_get_rotary()

    positions = torch.tensor([0, 1, 127, 1023, 8192], device="cuda")
    exponents = torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64
    angles = torch.outer(positions.float(), 1.0 / (10000 ** exponents))
    expected = torch.complex(angles.cos(), angles.sin())
    torch.testing.assert_close(model._freqs_cis_sliding[positions], expected, rtol=1e-6, atol=1e-6)
    unscaled_compress = torch.outer(positions.float(), 1.0 / (160000 ** exponents))
    assert not torch.allclose(
        model._freqs_cis_compress[positions], torch.complex(unscaled_compress.cos(), unscaled_compress.sin())
    )
    assert model.layers_infer[0].freqs_cis is model._freqs_cis_sliding
    for layer in model.layers_infer[1:]:
        assert layer.freqs_cis is model._freqs_cis_compress


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
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.load_way, model.tp_world_size_ = "HF", tp
    model.config = {"num_attention_heads": 64, "o_groups": 8, "index_n_heads": 64}
    model.args = SimpleNamespace(
        enable_dp_prefill_balance=balance,
        enable_ep_moe=ep,
        enable_tpsp_mix_mode=tpsp,
        enable_prefill_eplb=bool(redundancy),
    )
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
@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1)])
def test_output_projection_matches_bf16_reference(monkeypatch, tp_size, rank):
    from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight
    from lightllm.common.basemodel.layer_weights.meta_weights import platform_op
    from lightllm.models.deepseek_v4.layer_weights.transformer_layer_weight import DeepseekV4TransformerLayerWeight

    monkeypatch.setattr(
        platform_op,
        "get_env_start_args",
        lambda: SimpleNamespace(hardware_platform="cuda", enable_torch_fallback=False, enable_triton_fallback=False),
    )
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", str(tp_size))
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", str(rank))
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    torch.manual_seed(6102)
    layer = DeepseekV4TransformerLayerWeight.__new__(DeepseekV4TransformerLayerWeight)
    BaseLayerWeight.__init__(layer)
    layer.prefix, layer.data_type_ = "layers.0", torch.bfloat16
    layer.hidden, layer.n_heads, layer.head_dim = 128, 8, 64
    layer.q_lora_rank, layer.o_lora_rank, layer.o_groups = 128, 128, 2
    layer.get_quant_method = lambda name: None
    layer._init_qkvo()
    layer.experts = SimpleNamespace(quant_method=SimpleNamespace(weight_scale_suffix=None))

    weight_name = "layers.0.attn.wo_a.weight"
    scale_name = "layers.0.attn.wo_a.scale"
    packed = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.125, 0.25], [0.5, 1]], device="cuda").to(torch.float8_e8m0fnu)
    expected_weight = packed.float().unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)) * scales[:, None, :, None].float()
    expected_weight = expected_weight.flatten(2, 3).flatten(0, 1).bfloat16()
    weight_b = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    layer.load_hf_weights({weight_name: packed, scale_name: scales, "layers.0.attn.wo_b.weight": weight_b})
    assert layer.wo_a_.verify_load()
    local_groups = layer.o_groups // tp_size
    grouped_weight = expected_weight.view(2, 128, 256)[rank * local_groups : (rank + 1) * local_groups]
    heads = layer.n_heads // tp_size
    padded = torch.full((5, 64, 64), 9, device="cuda", dtype=torch.bfloat16)
    attention = padded[:, :heads]
    attention.normal_(std=0.1)
    angles = torch.randn(5, 32, device="cuda")
    cos, sin = angles.cos(), angles.sin()
    pairs = torch.view_as_complex(attention.float().reshape(5, heads, 32, 2).contiguous())
    rotated = torch.view_as_real(pairs * torch.complex(cos, sin).conj()[:, None]).flatten(-2).bfloat16()
    expected = torch.einsum("tgd,grd->tgr", rotated.reshape(5, local_groups, 256), grouped_weight).flatten(1)
    expected = expected @ weight_b[:, rank * local_groups * 128 : (rank + 1) * local_groups * 128].T

    infer_layer = DeepseekV4TransformerLayerInfer.__new__(DeepseekV4TransformerLayerInfer)
    infer_layer.qk_rope_head_dim, infer_layer.tp_groups, infer_layer.compress_ratio = 64, local_groups, 0
    infer_layer._tpsp_reduce = lambda input, infer_state: input
    state = SimpleNamespace(position_cos_sliding=cos, position_sin_sliding=sin)
    torch.testing.assert_close(infer_layer._get_o(attention, state, layer), expected, rtol=0.01, atol=0.01)
    assert padded[:, heads:].eq(9).all()

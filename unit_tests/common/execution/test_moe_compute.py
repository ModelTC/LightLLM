"""Native MoE contracts, independent routing/math, graph updates and TP loading."""

from dataclasses import replace, asdict
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.execution.moe import MoeBackend
from lightllm.common.execution.planner import Planner
from lightllm.common.execution.specs import DeviceCaps, FormatSpec, MoeSpec


def format_for(dtype, quantized=False, shard=(0, 1)):
    return FormatSpec(
        "fp8_block128" if quantized else "plain",
        torch.float8_e4m3fn if quantized else dtype,
        "expert_output_input",
        ("gate_up", "down", "gate_up_scale", "down_scale") if quantized else ("gate_up", "down"),
        1,
        shard,
        group_size=128 if quantized else None,
        scale_dtype=torch.float32 if quantized else None,
    )


def test_moe_contracts_reject_unsupported_semantics_and_formats():
    from lightllm.common.execution.model_adapter import grouped_moe_specs

    assert (
        grouped_moe_specs(dict(n_routed_experts=4, num_hidden_layers=2, first_k_dense_replace=2), torch.bfloat16) == {}
    )
    spec = MoeSpec(256, 256, 8, 2, torch.bfloat16)
    backend, fmt, caps = MoeBackend(), format_for(spec.dtype, True), DeviceCaps("cuda", 0, (9, 0))
    assert backend.probe(spec, fmt, caps).offers
    for changed, format, device, shared in (
        (spec, fmt, DeviceCaps("cpu"), 0),
        (spec, fmt, DeviceCaps("cuda", 0, (8, 0)), 0),
        (spec, replace(fmt, layout="input_output"), caps, 0),
        (spec, replace(fmt, group_size=64), caps, 0),
        (spec, replace(fmt, scale_dtype=torch.int32), caps, 0),
        (replace(spec, intermediate_size=130), fmt, caps, 0),
        (spec, fmt, caps, 1),
        (replace(spec, shared_experts=1, gated_shared=True), fmt, caps, 0),
    ):
        result = backend.probe(changed, format, device, fused_shared_experts=shared)
        assert result.reason and not result.offers
    for values in (
        dict(groups=3),
        dict(top_k=9),
        dict(group_score_topk=9),
        dict(scoring="other"),
        dict(gated_shared=True),
    ):
        with pytest.raises(ValueError):
            replace(spec, **values)


def route_reference(spec, logits, bias, scale, shared_gate, fused):
    scores = logits.float().sigmoid() if spec.scoring == "sigmoid" else logits.float().softmax(-1)
    selection = scores if bias is None else scores + bias.float()
    groups = selection.reshape(-1, spec.groups, spec.experts // spec.groups)
    chosen = groups.topk(spec.group_score_topk, -1).values.sum(-1).topk(spec.selected_groups, -1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(1, chosen[..., None].expand(-1, -1, groups.shape[-1]), True)
    ids = selection.masked_fill(~mask.flatten(1), -float("inf")).topk(spec.top_k, -1).indices
    weights = scores.gather(1, ids)
    if spec.renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    weights *= spec.routed_scale
    if scale is not None:
        weights *= scale[ids]
    if fused:
        ids = torch.cat(
            (ids, torch.arange(spec.experts, spec.experts + fused, device=ids.device).expand(len(ids), -1)), 1
        )
        shared = (
            torch.ones(len(logits), fused, device=logits.device)
            if shared_gate is None
            else shared_gate.float().sigmoid()
        )
        weights = torch.cat((weights, shared), 1)
    if spec.round_intermediates:
        weights = weights.to(spec.dtype)
    return weights, ids


def quantize_blocks(source):
    e, n, k = source.shape
    scale = source.reshape(e, n // 128, 128, k // 128, 128).abs().amax((2, 4)).clamp_min(1e-10) / 448
    expanded = scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
    encoded = (source / expanded).clamp(-448, 448).to(torch.float8_e4m3fn)
    return encoded, scale, encoded.float() * expanded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rounded", [False, True])
@torch.no_grad()
def test_moe_routed_shared_rounding_and_graph(dtype, rounded):
    spec = MoeSpec(
        128, 128, 2, 2, dtype, renormalize=True, shared_experts=1, gated_shared=True, round_intermediates=rounded
    )
    backend = MoeBackend()
    offer = backend.probe(spec, format_for(dtype), DeviceCaps("cuda", 0, (9, 0)), fused_shared_experts=1).offers[0]
    gate_up = torch.zeros(3, 256, 128, dtype=dtype, device="cuda")
    gate_up[:, :128, 0] = 32
    gate_up[:, 128:, 0] = 1
    base = 2048 if dtype == torch.float16 else 256
    down = torch.empty(3, 128, 128, dtype=dtype, device="cuda")
    down[0].fill_(2 * base / 4096)
    down[1].fill_(1 / 4096)
    down[2].fill_(-2 * base / 4096)
    bound = backend.bind(offer, gate_up, down)
    x = torch.ones(2, 128, dtype=dtype, device="cuda")
    logits = torch.zeros(2, 2, dtype=dtype, device="cuda")
    shared = torch.zeros(2, 1, dtype=dtype, device="cuda")
    # Routed outputs are base and 0.5, followed by a shared output of -base.
    # Rounding the routed subtotal loses 0.5 before the shared contribution is added.
    result = bound.run(x.clone(), logits, shared_gate=shared)
    torch.testing.assert_close(result, torch.full_like(x, 0 if rounded else 0.5), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = bound.run(x.clone(), logits, shared_gate=shared)
    x.fill_(2)
    graph.replay()
    torch.testing.assert_close(result, torch.full_like(x, 0 if rounded else 2), rtol=0, atol=0)

    logits.copy_(torch.tensor([[0.2, -0.3], [-0.7, 0.4]], device="cuda", dtype=dtype))
    shared.copy_(torch.tensor([[0.6], [-0.9]], device="cuda", dtype=dtype))
    weights, ids = bound.route(logits, shared)
    expected = logits.float().softmax(-1).gather(1, ids[:, :2].long())
    expected /= expected.sum(-1, keepdim=True)
    expected = torch.cat((expected, shared.float().sigmoid()), dim=-1)
    if rounded:
        expected = expected.to(dtype)
        torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(weights, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("round_intermediates", [False, True])
@torch.no_grad()
def test_moe_float32_preserves_projection_range_and_graph(round_intermediates):
    spec = MoeSpec(128, 128, 1, 1, torch.float32, round_intermediates=round_intermediates)
    backend = MoeBackend()
    offer = backend.probe(spec, format_for(spec.dtype), DeviceCaps("cuda", 0, (9, 0))).offers[0]
    gate_up = torch.ones(1, 256, 128, device="cuda")
    gate_up[:, 128:] *= 512
    down = torch.full((1, 128, 128), 1 / 128, device="cuda")
    bound = backend.bind(offer, gate_up, down)
    x, logits = torch.ones(1, 128, device="cuda"), torch.zeros(1, 1, device="cuda")
    # The up projection is 65536; every operation stays in the FP32 range.
    torch.testing.assert_close(bound.run(x.clone(), logits), torch.full_like(x, 2 ** 23), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = bound.run(x.clone(), logits)
    x.fill_(2)
    graph.replay()
    torch.testing.assert_close(output, torch.full_like(x, 2 ** 25), rtol=0, atol=0)


def quantized_activations(x):
    group = x.float().reshape(*x.shape[:-1], -1, 128)
    scale = group.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
    return ((group / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def expert_reference(spec, x, gate_up, down, weights, ids, quantized):
    # Mirror the specified storage rounding, with Torch matmul/activation rather than the MoE kernels.
    source = quantized_activations(x) if quantized else x.float()
    projections = torch.einsum("th,tknh->tkn", source, gate_up[ids]).to(x.dtype)
    gate, up = projections.chunk(2, -1)
    if spec.limit is not None:
        gate, up = gate.float().clamp(max=spec.limit), up.float().clamp(-spec.limit, spec.limit)
        if spec.clamp_up_add_one:
            up = up + 1
    gate = (gate.float() * torch.sigmoid(gate.float() * spec.alpha)).to(x.dtype)
    activated = (gate * up).to(x.dtype)
    source = quantized_activations(activated) if quantized else activated.float()
    expert_out = torch.einsum("tki,tkhi->tkh", source, down[ids])
    if spec.round_intermediates:
        expert_out = expert_out.to(x.dtype)
        weighted = (expert_out * weights[..., None]).to(x.dtype)
        routed = weighted[:, : spec.top_k].float().sum(1).to(x.dtype)
        shared = weighted[:, spec.top_k :].float().sum(1).to(x.dtype)
        return routed + shared
    return (expert_out * weights[..., None]).to(x.dtype).float().sum(1).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mode", ["softmax", "group_max", "group_top2", "shared_gate", "shared_gate_rounded", "clamp", "clamp_add_one"]
)
@pytest.mark.parametrize("quantized", [False, True])
@torch.no_grad()
def test_moe_native_routing_math_capture_and_graph(dtype, mode, quantized, monkeypatch):
    torch.manual_seed(181)
    spec = MoeSpec(
        256,
        256,
        8,
        2,
        dtype,
        renormalize=mode != "softmax",
        routed_scale=1.7,
        shared_experts=1 if mode == "group_max" or mode.startswith("shared_gate") else 0,
        gated_shared=mode.startswith("shared_gate"),
        round_intermediates=mode == "shared_gate_rounded",
    )
    if mode in ("group_max", "group_top2"):
        spec = replace(
            spec, groups=2, selected_groups=1, scoring="sigmoid", group_score_topk=2 if mode == "group_top2" else 1
        )
    if mode.startswith("clamp"):
        spec = replace(
            spec, limit=0.2, alpha=1.3 if mode == "clamp_add_one" else 1, clamp_up_add_one=mode == "clamp_add_one"
        )
    count = spec.experts + spec.shared_experts
    a, b = (torch.randn(shape, device="cuda", dtype=dtype) * 0.08 for shape in ((count, 512, 256), (count, 256, 256)))
    a_scale = b_scale = None
    ref_a, ref_b = a.float(), b.float()
    if quantized:
        a, a_scale, ref_a = quantize_blocks(ref_a)
        b, b_scale, ref_b = quantize_blocks(ref_b)
    bias = torch.randn(8, device="cuda") * 0.1 if mode.startswith("group") else None
    expert_scale = torch.rand(8, device="cuda") + 0.5 if mode == "group_top2" else None
    fmt, backend = format_for(dtype, quantized), MoeBackend()
    offer = backend.probe(spec, fmt, DeviceCaps("cuda", 0, (9, 0)), fused_shared_experts=spec.shared_experts).offers[0]
    bound = backend.bind(
        offer, a, b, gate_up_scale=a_scale, down_scale=b_scale, correction_bias=bias, expert_scale=expert_scale
    )
    x = torch.randn(17, 256, device="cuda", dtype=dtype) * 0.2
    logits = torch.randn(17, 8, device="cuda", dtype=torch.float32)
    shared = torch.randn(17, 1, device="cuda", dtype=dtype) if spec.gated_shared else None
    captured = torch.empty(17, 2, device="cuda", dtype=torch.int64)

    def capture(ids):
        assert ids.is_contiguous() and ids.shape == (17, 2)
        captured.copy_(ids)

    def check_routes():
        expected_weights, expected_ids = route_reference(spec, logits, bias, expert_scale, shared, spec.shared_experts)
        weights, ids = bound.route(logits, shared)
        order = ids.argsort(1)
        ref_order = expected_ids.argsort(1)
        torch.testing.assert_close(ids.gather(1, order).long(), expected_ids.gather(1, ref_order), rtol=0, atol=0)
        torch.testing.assert_close(
            weights.gather(1, order), expected_weights.gather(1, ref_order), rtol=1e-5, atol=1e-6
        )
        return expert_reference(spec, x, ref_a, ref_b, expected_weights, expected_ids, quantized), expected_ids[:, :2]

    from lightllm.common.basemodel.triton_kernel.fused_moe import moe_silu_and_mul

    monkeypatch.setattr(
        moe_silu_and_mul, "ffn_use_tanh_approximate_gelu", lambda: pytest.fail("runtime queried model config")
    )

    def check(actual, expected):
        error = actual.float() - expected.float()
        assert error.square().mean().sqrt() / expected.float().square().mean().sqrt() < 0.01
        assert error.abs().max() < expected.abs().max() / 32

    expected, expected_ids = check_routes()
    out = x.clone()
    assert bound.run(out, logits, shared_gate=shared, capture=capture) is out
    check(out, expected)
    torch.testing.assert_close(captured.sort(1).values, expected_ids.sort(1).values, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(x)
        bound.run(out, logits, shared_gate=shared, capture=capture)
    x.add_(0.03)
    logits.mul_(-1)
    if shared is not None:
        shared.add_(1)
    expected, expected_ids = check_routes()
    graph.replay()
    check(out, expected)
    torch.testing.assert_close(captured.sort(1).values, expected_ids.sort(1).values, rtol=0, atol=0)
    assert bound.run(x[:0], logits[:0], shared_gate=None if shared is None else shared[:0]).numel() == 0
    with pytest.raises(ValueError, match="input"):
        bound.run(x[:, ::2], logits, shared_gate=shared)
    with pytest.raises(ValueError, match="view"):
        backend.bind(offer, a[:, :, :128], b, gate_up_scale=a_scale, down_scale=b_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("method", ["none", "fp8w8a8-b128-triton"])
@pytest.mark.parametrize("shared", [False, True])
@torch.no_grad()
def test_moe_planner_binds_tp_weights_and_survives_loader_replacement(method, shared, monkeypatch):
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
    from lightllm.common.quantization import QUANTMETHODS
    from lightllm.common.quantization.quantize_method import WeightPack
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_redundancy_expert_num, get_redundancy_expert_ids

    for key, value in {
        "CURRENT_DEVICE_ID": 0,
        "CURRENT_RANK_IN_DP": 1,
        "DP_WORLD_SIZE": 2,
        "GLOBAL_RANK": 1,
        "GLOBAL_WORLD_SIZE": 2,
    }.items():
        monkeypatch.setenv("LIGHTLLM_" + key, str(value))
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(StartArgs())))
    for cached in (get_env_start_args, get_redundancy_expert_num, get_redundancy_expert_ids):
        cached.cache_clear()
    torch.manual_seed(191)
    spec = MoeSpec(256, 512, 4, 2, torch.bfloat16, renormalize=True)
    config = dict(num_experts_per_tok=2, norm_topk_prob=True)
    weight = FusedMoeWeight(
        "gate_proj",
        "down_proj",
        "up_proj",
        "",
        "experts",
        4,
        256,
        512,
        spec.dtype,
        quant_method=QUANTMETHODS.get(method),
        layer_num=3,
        network_config=config,
    )
    assert weight.tp_rank_ == 1 and weight.tp_world_size_ == 2
    layer, recipe = SimpleNamespace(layer_num_=3, experts=weight), SimpleNamespace(layer_num_=3, is_moe=True)
    with pytest.raises(ValueError, match="fully loaded"):
        Planner.bind_moe({3: spec}, [layer], [recipe])
    weights = {}
    for i in range(4):
        for name, shape in (("gate_proj", (512, 256)), ("up_proj", (512, 256)), ("down_proj", (256, 512))):
            weights[f"experts.{i}.{name}.weight"] = torch.randn(shape, dtype=spec.dtype) * 0.08
    weight.load_hf_weights(weights)
    x = torch.randn(7, 256, device="cuda", dtype=spec.dtype)
    logits = torch.randn(7, 4, device="cuda")
    expected = weight.experts(x.clone(), logits, 2, True, False, 0, 0)
    recipes = [recipe, SimpleNamespace(layer_num_=9, is_moe=True)] if shared else [recipe]
    plan = Planner.bind_moe({r.layer_num_: spec for r in recipes}, [layer] * len(recipes), recipes)
    assert len(plan.bindings) == len(recipes) and not plan.legacy
    for item in recipes:
        assert item.bound_moe.gate_up.data_ptr() == weight.w13.weight.data_ptr()
    bound = recipe.bound_moe
    assert bound.gate_up.data_ptr() == weight.w13.weight.data_ptr()
    if method != "none":
        decoded = []
        for pack in (weight.w13, weight.w2):
            decoded.append(pack.weight.float() * pack.weight_scale.repeat_interleave(128, 1).repeat_interleave(128, 2))
        routing_weights, routing_ids = route_reference(spec, logits, None, None, None, 0)
        expected = expert_reference(spec, x, *decoded, routing_weights, routing_ids, True)
    weight.w13 = WeightPack(weight=torch.zeros_like(weight.w13.weight))
    weight.w2 = WeightPack(weight=torch.zeros_like(weight.w2.weight))
    weight.fuse_moe_impl = lambda *a, **kw: pytest.fail("native MoE called the old weight executor")
    actual = bound.run(x, logits)
    if method == "none":
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        error = actual.float() - expected.float()
        assert error.square().mean().sqrt() / expected.float().square().mean().sqrt() < 0.01
        assert error.abs().max() < expected.abs().max() / 32
    for cached in (get_env_start_args, get_redundancy_expert_num, get_redundancy_expert_ids):
        cached.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_score_topk", [1, 2])
@torch.no_grad()
def test_moe_non_power_of_two_experts_and_concurrent_graphs(group_score_topk):
    torch.manual_seed(193)
    spec = MoeSpec(
        128,
        128,
        160,
        6,
        torch.bfloat16,
        renormalize=True,
        scoring="sigmoid",
        groups=8,
        selected_groups=3,
        group_score_topk=group_score_topk,
    )
    backend, fmt = MoeBackend(), format_for(spec.dtype)
    offer = backend.probe(spec, fmt, DeviceCaps("cuda", 0, (9, 0))).offers[0]
    a = torch.randn(160, 256, 128, device="cuda", dtype=spec.dtype) * 0.04
    b = torch.randn(160, 128, 128, device="cuda", dtype=spec.dtype) * 0.04
    bias = torch.rand(160, device="cuda") * 0.1 - 0.5
    bound = backend.bind(offer, a, b, correction_bias=bias)
    lanes = []
    for size in (3, 5):
        x = torch.randn(size, 128, device="cuda", dtype=spec.dtype)
        logits = torch.randn(size, 160, device="cuda")
        out = x.clone()
        bound.run(out, logits)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out.copy_(x)
            bound.run(out, logits)
        x.add_(0.1)
        logits.neg_()
        weights, ids = route_reference(spec, logits, bias, None, None, 0)
        actual_weights, actual_ids = bound.route(logits)
        order, actual_order = ids.argsort(-1), actual_ids.argsort(-1)
        torch.testing.assert_close(ids.gather(1, order), actual_ids.gather(1, actual_order).long(), rtol=0, atol=0)
        torch.testing.assert_close(
            weights.gather(1, order), actual_weights.gather(1, actual_order), rtol=1e-5, atol=1e-6
        )
        expected = expert_reference(spec, x, a.float(), b.float(), weights, ids, False)
        lanes.append((graph, x, logits, out, expected))
    streams = [torch.cuda.Stream() for _ in lanes]
    for stream, (graph, _, _, _, _) in zip(streams, lanes):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
    for stream, (_, _, _, out, expected) in zip(streams, lanes):
        torch.cuda.current_stream().wait_stream(stream)
        error = out.float() - expected.float()
        assert error.square().mean().sqrt() / expected.float().square().mean().sqrt() < 0.01
        assert error.abs().max() < expected.abs().max() / 32

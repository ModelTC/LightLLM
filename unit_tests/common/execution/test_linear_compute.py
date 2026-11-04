"""Bound matrix execution, independent quantized references, and loading/TP separation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.core.compute.linear.backend import LinearBackend
from lightllm.core.planner.planner import Planner
from lightllm.core.contracts.device import DeviceCaps
from lightllm.core.contracts.state import FormatSpec
from lightllm.core.compute.linear.spec import LinearSpec


def contracts(name, dtype=torch.bfloat16, batches=None):
    backend = LinearBackend(name)
    encoding = backend.encodings[name]
    quantized = encoding != "plain"
    spec = LinearSpec(256, 256, dtype, batches)
    fmt = FormatSpec(
        encoding,
        torch.float8_e4m3fn if quantized else dtype,
        "batch_input_output" if batches is not None else "output_input",
        ("weight", "scale") if quantized else ("weight",),
        1,
        group_size=128 if encoding == "fp8_block128" else None,
        scale_dtype=torch.float32 if quantized else None,
    )
    return backend, spec, fmt


def test_linear_probe_rejects_incompatible_compute_and_storage():
    backend, spec, fmt = contracts("triton.linear.fp8_block128")
    caps = DeviceCaps("cuda", 0, (9, 0))
    assert backend.probe(spec, fmt, caps).offers
    for changed, format, device in (
        (replace(spec, in_features=129), fmt, caps),
        (replace(spec, out_features=129), fmt, caps),
        (replace(spec, batches=2), fmt, caps),
        (spec, replace(fmt, scale_dtype=torch.float16), caps),
        (spec, fmt, DeviceCaps("cuda", 0, (8, 0))),
        (spec, fmt, DeviceCaps("cpu")),
        (spec, replace(fmt, encoding="fp8_per_tensor"), caps),
        (spec, replace(fmt, group_size=64), caps),
    ):
        result = backend.probe(changed, format, device)
        assert result.reason and not result.offers


def test_linear_binding_rejects_bad_views_before_execution():
    backend, spec, fmt = contracts("torch.linear", torch.float32)
    offer = backend.probe(spec, fmt, DeviceCaps("cpu")).offers[0]
    with pytest.raises(ValueError, match="weight view"):
        backend.bind(offer, torch.zeros(128, 256))
    with pytest.raises(ValueError, match="bias"):
        backend.bind(offer, torch.zeros(256, 256), bias=torch.zeros(1))
    bound = backend.bind(offer, torch.eye(256))
    x = torch.randn(2, 256)
    torch.testing.assert_close(bound.run(x), x, rtol=0, atol=0)
    with pytest.raises(ValueError, match="input"):
        bound.run(torch.zeros(2, 128))
    with pytest.raises(ValueError, match="output"):
        bound.run(x, torch.empty(2, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["bias", "strided", "fp32_out", "batched"])
@torch.no_grad()
def test_plain_linear_views_and_graph(dtype, mode):
    torch.manual_seed(163)
    backend, spec, fmt = contracts("torch.linear", dtype, 3 if mode == "batched" else None)
    if spec.batches is not None:
        weight = torch.randn(3, 256, 256, device="cuda", dtype=dtype) * 0.1
        x = torch.randn(3, 17, 256, device="cuda", dtype=dtype) * 0.2
    else:
        weight = torch.randn(256, 256, device="cuda", dtype=dtype) * 0.1
        x = torch.randn(17, 512, device="cuda", dtype=dtype)[:, ::2] * 0.2
    if mode == "strided":
        weight = weight.t()
        x = torch.randn(34, 256, device="cuda", dtype=dtype)[::2]
    bias = torch.randn(256, device="cuda", dtype=dtype) * 0.1 if mode == "bias" else None
    offer = backend.probe(spec, fmt, DeviceCaps("cuda", 0, (9, 0))).offers[0]
    bound = backend.bind(offer, weight, bias=bias)
    out = torch.empty((*x.shape[:-1], 256), device="cuda", dtype=torch.float32 if mode == "fp32_out" else dtype)

    def reference():
        result = x.float() @ (weight.float() if mode == "batched" else weight.float().t())
        return (result if bias is None else result + bias.float()).to(out.dtype)

    assert bound.run(x, out) is out
    torch.testing.assert_close(out, reference(), rtol=0.01, atol=0.004)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bound.run(x, out)
    x.add_(0.1)
    weight.mul_(0.9)
    graph.replay()
    torch.testing.assert_close(out, reference(), rtol=0.01, atol=0.004)
    empty = x[:, :0] if mode == "batched" else x[:0]
    assert bound.run(empty).numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "name",
    [
        "triton.linear.fp8_tensor",
        "sgl.linear.fp8_tensor",
        "triton.linear.fp8_channel_g64",
        "triton.linear.fp8_channel_g128",
        "triton.linear.fp8_block128",
    ],
)
@torch.no_grad()
def test_fp8_linear_matches_independent_quantization_and_graph(name, dtype):
    torch.manual_seed(167)
    backend, spec, fmt = contracts(name, dtype)
    source = torch.randn(256, 256, device="cuda") * 0.1
    if fmt.encoding == "fp8_block128":
        groups = source.reshape(2, 128, 2, 128)
        scale = groups.abs().amax((1, 3)) / 448
        expanded = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    elif fmt.encoding == "fp8_per_channel":
        scale = source.abs().amax(1) / 448
        expanded = scale[:, None]
    else:
        scale = source.abs().amax().reshape(1) / 448
        expanded = scale
        if name.startswith("sgl."):
            scale = scale.expand(256).clone()
    weight = (source / expanded).clamp(-448, 448).to(torch.float8_e4m3fn)
    decoded = weight.float() * expanded
    offer = backend.probe(spec, fmt, DeviceCaps("cuda", 0, (9, 0))).offers[0]
    bound = backend.bind(offer, weight, scale)
    x = torch.randn(17, 256, device="cuda", dtype=dtype) * 0.2
    out = torch.empty(17, 256, device="cuda", dtype=dtype)
    group = 64 if name.endswith("g64") else 128 if "channel" in name or "block128" in name else 256

    def check():
        groups = x.float().reshape(17, -1, group)
        scales = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
        encoded = (groups / scales).clamp(-448, 448).to(torch.float8_e4m3fn)
        expected = (encoded.float() * scales).reshape(17, 256) @ decoded.t()
        error = out.float() - expected
        assert error.square().mean().sqrt() / expected.square().mean().sqrt() < 0.01
        assert error.abs().max() < expected.abs().max() / 64

    assert bound.run(x, out) is out
    check()
    torch.testing.assert_close(bound.run(x), out, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bound.run(x, out)
    x.add_(0.03)
    graph.replay()
    check()
    assert bound.run(x[:0]).shape == (0, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("method", ["none", "fp8w8a8-pt-triton", "fp8w8a8g128-triton", "fp8w8a8-b128-triton"])
@torch.no_grad()
def test_linear_planner_binds_loaded_tp_views_without_loader_calls(method, monkeypatch):
    from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.rowmm_weight import ROWMMWeight
    from lightllm.common.quantization import QUANTMETHODS
    from lightllm.common.quantization.quantize_method import WeightPack

    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    torch.manual_seed(173)
    quant = QUANTMETHODS.get(method)
    weight = ROWMMWeight(
        128, [256, 256], ["gate.weight", "up.weight"], torch.bfloat16, quant_method=quant, tp_rank=1, tp_world_size=2
    )
    layer = SimpleNamespace(layer_num_=2, projection=weight)
    gate, up = (torch.randn(256, 128, dtype=torch.bfloat16) * 0.1 for _ in range(2))
    weight.load_hf_weights({"gate.weight": gate})
    with pytest.raises(ValueError, match="fully loaded"):
        Planner.bind_linears((layer,), torch.empty)
    weight.load_hf_weights({"up.weight": up})
    x = torch.randn(5, 128, dtype=torch.bfloat16, device="cuda") * 0.2
    expected = weight.mm(x, use_custom_tensor_mananger=False)
    plan = Planner.bind_linears((layer,), torch.empty)
    assert len(plan.bindings) == 1 and not plan.legacy
    bound = plan.bindings[0][1]
    assert bound.weight.data_ptr() == weight.mm_param.weight.data_ptr()
    assert weight.mm.__self__ is bound
    monkeypatch.setattr(quant, "apply", lambda *a, **kw: pytest.fail("compute queried weight loader"))
    weight.mm_param = WeightPack(weight=torch.zeros_like(weight.mm_param.weight))
    actual = weight.mm(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

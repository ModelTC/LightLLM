from lightllm.platform import get_device_runtime
import pytest
import torch

from lightllm.common.attention.api import CapabilityResult, VarlenAttentionInputs
from lightllm.common.attention.selection import get_varlen_backend_class
from lightllm.common.attention.varlen import AutoVarlenAttention


def make_inputs(lengths=(3, 0, 5), *, device="cpu", dtype=torch.float32, head_dim=80, strided=False):
    tokens = sum(lengths)
    if strided:
        qkv = torch.randn(tokens, 3, 2, head_dim, dtype=dtype, device=device)
        q, k, v = qkv.unbind(1)
    else:
        q, k, v = [torch.randn(tokens, 2, head_dim, dtype=dtype, device=device) for _ in range(3)]
    cu = torch.tensor([0, *lengths], dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    return VarlenAttentionInputs(q, k, v, cu, max(lengths))


def reference_attention(inputs):
    out = torch.empty_like(inputs.query)
    offsets = inputs.cu_seqlens.cpu().tolist()
    for start, end in zip(offsets, offsets[1:]):
        if start == end:
            continue
        q, k, v = [x[start:end].float().transpose(0, 1) for x in (inputs.query, inputs.key, inputs.value)]
        scores = q @ k.transpose(-1, -2) * inputs.query.shape[-1] ** -0.5
        out[start:end] = (scores.softmax(-1) @ v).transpose(0, 1).to(out.dtype)
    return out


@pytest.mark.parametrize("error_type", [None, RuntimeError, TypeError, AssertionError])
def test_audio_fallback_overwrites_partial_output(monkeypatch, error_type):
    from lightllm.common.attention import varlen

    calls = []

    class FakeBackend:
        @classmethod
        def check_environment(cls, caps):
            return CapabilityResult.ok()

        @classmethod
        def check_support(cls, inputs, caps=None):
            return CapabilityResult.ok()

        def forward(self, inputs, *, out):
            calls.append("fa3")
            if error_type:
                out[0].fill_(99)
                raise error_type("shape-dependent FA3 failure")
            out.fill_(9)
            return out

    class Fallback(FakeBackend):
        def forward(self, inputs, *, out):
            calls.append("triton")
            out.fill_(7)
            return out

    monkeypatch.setattr(varlen, "get_varlen_backend_class", lambda name: {"fa3": FakeBackend, "triton": Fallback}[name])
    backend = AutoVarlenAttention()
    inputs = make_inputs()
    out = torch.full_like(inputs.query, -123)
    assert backend.forward(inputs, out=out) is out
    assert torch.equal(out, torch.full_like(out, 7 if error_type else 9))
    out.fill_(-123)
    backend.forward(inputs, out=out)
    assert calls == (["fa3", "triton", "triton"] if error_type else ["fa3", "fa3"])
    assert torch.equal(out, torch.full_like(out, 7 if error_type else 9))


@pytest.mark.parametrize("lengths", [(3, 0, 5), (1, 7), (0, 0)])
def test_sdpa_preserves_sequence_boundaries_and_output_storage(lengths):
    inputs = make_inputs(lengths)
    inputs.query.zero_()
    inputs.key.zero_()
    start = 0
    for i, length in enumerate(lengths):
        inputs.value[start : start + length].fill_(i + 1)
        start += length
    out = torch.empty_like(inputs.query)
    result = get_varlen_backend_class("sdpa")().forward(inputs, out=out)
    assert result is out
    torch.testing.assert_close(out, inputs.value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["fa3", "triton", "sdpa", "xformers"])
@pytest.mark.parametrize("dtype,atol", [(torch.float16, 1e-2), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("head_dim,strided", [(64, False), (80, True), (128, False)])
def test_varlen(provider, dtype, atol, head_dim, strided):
    backend = get_varlen_backend_class(provider)
    environment = backend.check_environment(get_device_runtime().caps)
    if not environment.supported:
        pytest.skip(environment.reason)
    torch.manual_seed(0)
    inputs = make_inputs((1, 0, 65, 129), device="cuda", dtype=dtype, head_dim=head_dim, strided=strided)
    storage = torch.full((*inputs.query.shape[:2], head_dim * 2), -123, device="cuda", dtype=dtype)
    out = storage[..., :head_dim]
    before = [x.clone() for x in (inputs.query, inputs.key, inputs.value)]
    result = backend().forward(inputs, out=out)
    assert result is out
    torch.testing.assert_close(out, reference_attention(inputs), atol=atol, rtol=0)
    assert torch.equal(storage[..., head_dim:], torch.full_like(out, -123))
    for actual, expected in zip((inputs.query, inputs.key, inputs.value), before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["fa3", "triton", "sdpa", "xformers"])
def test_empty_batch(provider):
    backend = get_varlen_backend_class(provider)
    environment = backend.check_environment(get_device_runtime().caps)
    if not environment.supported:
        pytest.skip(environment.reason)
    inputs = make_inputs((0, 0), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(inputs.query)
    assert backend().forward(inputs, out=out) is out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["fa3", "triton"])
def test_graph_replay_uses_new_inputs(provider):
    backend_cls = get_varlen_backend_class(provider)
    environment = backend_cls.check_environment(get_device_runtime().caps)
    if not environment.supported:
        pytest.skip(environment.reason)
    backend = backend_cls()
    assert backend.supports_graph
    inputs = make_inputs((3, 5), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(inputs.query)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        backend.forward(inputs, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.forward(inputs, out=out)
    inputs.value.add_(1)
    inputs.cu_seqlens.copy_(torch.tensor([0, 5, 8], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(out, reference_attention(inputs), atol=2e-2, rtol=0)

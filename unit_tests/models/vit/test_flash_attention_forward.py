import pytest
import torch

from lightllm.common.attention.api import UnsupportedAttention, VarlenAttentionInputs
from lightllm.common.attention.registry import get_varlen_attention_op_class
from lightllm.common.attention.varlen import AutoVarlenAttentionRunner
from lightllm.platform import get_device_runtime


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


@pytest.mark.parametrize("error_type", [None, UnsupportedAttention])
def test_audio_fallback_only_rejects_current_input(monkeypatch, error_type):
    from lightllm.common.attention import varlen

    calls = []

    class FakeOp:
        @classmethod
        def check_environment(cls, caps):
            return None

        @classmethod
        def check_support(cls, inputs, caps=None):
            calls.append("fa3")
            if error_type:
                raise error_type("shape-dependent FA3 failure")

        def forward(self, inputs, *, out):
            out.fill_(9)
            return out

    class Fallback(FakeOp):
        @classmethod
        def check_support(cls, inputs, caps=None):
            return None

        def forward(self, inputs, *, out):
            calls.append("triton")
            out.fill_(7)
            return out

    monkeypatch.setattr(varlen, "get_varlen_attention_op_class", lambda name: {"fa3": FakeOp, "triton": Fallback}[name])
    op = AutoVarlenAttentionRunner()
    inputs = make_inputs()
    out = torch.full_like(inputs.query, -123)
    assert op.forward(inputs, out=out) is out
    assert torch.equal(out, torch.full_like(out, 7 if error_type else 9))
    out.fill_(-123)
    op.forward(inputs, out=out)
    assert calls == (["fa3", "triton", "fa3", "triton"] if error_type else ["fa3", "fa3"])
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
    result = get_varlen_attention_op_class("sdpa")().forward(inputs, out=out)
    assert result is out
    torch.testing.assert_close(out, inputs.value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["fa3", "triton", "sdpa", "xformers"])
@pytest.mark.parametrize("dtype,atol", [(torch.float16, 1e-2), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("head_dim,strided", [(64, False), (80, True), (128, False)])
def test_varlen(provider, dtype, atol, head_dim, strided):
    op = get_varlen_attention_op_class(provider)
    try:
        op.check_environment(get_device_runtime().caps)
    except UnsupportedAttention as exc:
        pytest.skip(str(exc))
    torch.manual_seed(0)
    inputs = make_inputs((1, 0, 65, 129), device="cuda", dtype=dtype, head_dim=head_dim, strided=strided)
    storage = torch.full((*inputs.query.shape[:2], head_dim * 2), -123, device="cuda", dtype=dtype)
    out = storage[..., :head_dim]
    before = [x.clone() for x in (inputs.query, inputs.key, inputs.value)]
    result = op().forward(inputs, out=out)
    assert result is out
    torch.testing.assert_close(out, reference_attention(inputs), atol=atol, rtol=0)
    assert torch.equal(storage[..., head_dim:], torch.full_like(out, -123))
    for actual, expected in zip((inputs.query, inputs.key, inputs.value), before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["fa3", "triton", "sdpa", "xformers"])
def test_empty_batch(provider):
    op = get_varlen_attention_op_class(provider)
    try:
        op.check_environment(get_device_runtime().caps)
    except UnsupportedAttention as exc:
        pytest.skip(str(exc))
    inputs = make_inputs((0, 0), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(inputs.query)
    assert op().forward(inputs, out=out) is out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("provider", ["fa3", "triton"])
def test_graph_replay_uses_new_inputs(provider):
    op_cls = get_varlen_attention_op_class(provider)
    try:
        op_cls.check_environment(get_device_runtime().caps)
    except UnsupportedAttention as exc:
        pytest.skip(str(exc))
    op = op_cls()
    assert op.supports_graph
    inputs = make_inputs((3, 5), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(inputs.query)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.forward(inputs, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.forward(inputs, out=out)
    inputs.value.add_(1)
    inputs.cu_seqlens.copy_(torch.tensor([0, 5, 8], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(out, reference_attention(inputs), atol=2e-2, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_audio_attention_matches_projected_qkv(monkeypatch):
    from lightllm.models.qwen3_omni_moe_thinker import qwen3_omni_audio as audio

    torch.manual_seed(19)
    attention = audio.Qwen3OmniMoeAudioAttention(128, 2, 0).cuda().to(torch.bfloat16).eval()
    monkeypatch.setattr(
        audio.g_cache_manager,
        "alloc_tensor",
        lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device),
    )
    hidden = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    cu = torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        q, k, v = [proj(hidden).reshape(8, 2, 64) for proj in (attention.q_proj, attention.k_proj, attention.v_proj)]
        expected = attention.out_proj(reference_attention(VarlenAttentionInputs(q, k, v, cu, 5)).reshape(8, 128))
        actual = attention(hidden, cu_seqlens=cu, max_seqlen=5)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("error_type", [RuntimeError, TypeError, AssertionError])
def test_audio_kernel_errors_propagate_without_retry(monkeypatch, error_type):
    from lightllm.common.attention import varlen

    class Broken:
        @classmethod
        def check_environment(cls, caps):
            pass

        @classmethod
        def check_support(cls, inputs, caps=None):
            pass

        def forward(self, inputs, *, out):
            out.fill_(99)
            raise error_type("kernel failed")

    calls = []

    def load(name):
        calls.append(name)
        return Broken

    monkeypatch.setattr(varlen, "get_varlen_attention_op_class", load)
    op = AutoVarlenAttentionRunner()
    inputs = make_inputs()
    with pytest.raises(error_type, match="kernel failed"):
        op.forward(inputs, out=torch.empty_like(inputs.query))
    assert calls == ["fa3"]
    assert not op.unavailable


def test_audio_encoder_shares_runner_only_with_its_layers():
    from lightllm.models.qwen3_omni_moe_thinker.qwen3_omni_audio import Qwen3OmniMoeAudioEncoder

    def encoder():
        return Qwen3OmniMoeAudioEncoder(
            {},
            d_model=16,
            encoder_attention_heads=2,
            encoder_layers=2,
            encoder_ffn_dim=32,
            downsample_hidden_size=4,
            output_dim=16,
        )

    first, second = encoder(), encoder()
    assert first.layers[0].self_attn.attention_op is first.layers[1].self_attn.attention_op
    assert first.layers[0].self_attn.attention_op is not second.layers[0].self_attn.attention_op


@pytest.mark.parametrize("boundaries,maximum", [([1, 3, 8], 5), ([0, 7, 6, 8], 7), ([0, 3, 7], 5), ([0, 3, 8], 4)])
def test_varlen_rejects_invalid_boundaries(boundaries, maximum):
    inputs = make_inputs()._replace(cu_seqlens=torch.tensor(boundaries, dtype=torch.int32), max_seqlen=maximum)
    with pytest.raises(ValueError, match="cu_seqlens"):
        inputs.validate()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_auto_validates_once_per_call_and_rechecks_mutated_boundaries(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    inputs = make_inputs(device=device)
    original = VarlenAttentionInputs.validate
    calls = []

    def validate(self):
        calls.append(self.cu_seqlens)
        return original(self)

    monkeypatch.setattr(VarlenAttentionInputs, "validate", validate)
    op = AutoVarlenAttentionRunner(providers=("sdpa",))
    output = torch.empty_like(inputs.query)
    op.forward(inputs, out=output)
    inputs.cu_seqlens.copy_(torch.tensor([0, 1, 1, 8], dtype=torch.int32, device=device))
    op.forward(inputs._replace(max_seqlen=7), out=output)
    assert len(calls) == 2
    torch.testing.assert_close(output, reference_attention(inputs._replace(max_seqlen=7)))


@pytest.mark.parametrize("provider", ["fa3", "triton", "xformers", "sdpa"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "overlap", "alias", "head_stride"])
def test_varlen_output_validation(provider, invalid):
    inputs = make_inputs()
    out = torch.empty_like(inputs.query)
    if invalid == "shape":
        out = out[:-1]
    elif invalid == "dtype":
        out = out.double()
    elif invalid == "device":
        out = out.to("meta")
    elif invalid == "overlap":
        out = out[:1].expand_as(out)
    elif invalid == "alias":
        out = inputs.key
    else:
        out = torch.empty((*out.shape[:-1], out.shape[-1] * 2))[..., ::2]
    with pytest.raises(ValueError, match="varlen out"):
        get_varlen_attention_op_class(provider)().forward(inputs, out=out)

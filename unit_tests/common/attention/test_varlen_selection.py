import pytest

from lightllm.common.attention import selection
from lightllm.common.attention.api import UnsupportedAttention
from lightllm.utils import backend_validator


def test_auto_probes_actual_varlen_path_on_assigned_device(monkeypatch):
    calls = []

    def probe(name, **kwargs):
        calls.append((name, kwargs))
        return name == "sdpa"

    monkeypatch.setattr(backend_validator, "_validate", probe)
    assert selection.select_varlen_attention_backend(device=3) == "sdpa"
    assert calls == [
        (name, {"varlen": True, "device": 3, "dtype": None, "head_dim": 80}) for name in ("fa3", "xformers", "sdpa")
    ]


@pytest.mark.parametrize("requested", ["auto", "fa3", "triton", "missing"])
def test_rejected_backends_are_not_returned(monkeypatch, requested):
    calls = []

    def probe(name, **kwargs):
        calls.append(name)
        return False

    monkeypatch.setattr(backend_validator, "_validate", probe)
    with pytest.raises(selection.BackendSelectionError) as error:
        selection.select_varlen_attention_backend(requested)
    expected = list(selection.varlen_backends) if requested == "auto" else [requested]
    assert list(error.value.failures) == expected
    assert calls == ([] if requested == "missing" else expected)


def test_varlen_probe_exercises_selected_backend(monkeypatch):
    from lightllm.common.attention.backends.dense.sdpa.varlen import SdpaVarlenAttentionBackend

    monkeypatch.setattr(backend_validator.torch.cuda, "is_available", lambda: False)
    assert backend_validator._validate_varlen("sdpa") == (True, None)
    monkeypatch.setattr(SdpaVarlenAttentionBackend, "forward", lambda self, inputs, out: out.fill_(float("nan")))
    success, reason = backend_validator._validate_varlen("sdpa")
    assert not success
    assert "mismatch" in reason


@pytest.mark.parametrize("check", ["check_environment", "check_support"])
def test_rejected_probe_does_not_execute_backend(monkeypatch, check):
    from lightllm.common.attention.backends.dense.sdpa.varlen import SdpaVarlenAttentionBackend

    monkeypatch.setattr(backend_validator.torch.cuda, "is_available", lambda: False)

    def reject(cls, *args):
        raise UnsupportedAttention("layout")

    monkeypatch.setattr(SdpaVarlenAttentionBackend, check, classmethod(reject))
    monkeypatch.setattr(
        SdpaVarlenAttentionBackend, "forward", lambda *args, **kwargs: pytest.fail("rejected backend ran")
    )
    assert backend_validator._validate_varlen("sdpa") == (False, "layout")


def test_worker_initialization_uses_actual_inputs_and_removes_dispatch(monkeypatch):
    import torch

    from lightllm.common.attention.api import VarlenAttentionInputs
    from lightllm.common.attention.backends.dense.sdpa.varlen import SdpaVarlenAttentionBackend
    from lightllm.server import visualserver

    calls = []

    def select(requested, **kwargs):
        calls.append((requested, kwargs))
        return "sdpa"

    monkeypatch.setattr(visualserver, "select_varlen_attention_backend", select)
    monkeypatch.setattr(visualserver, "VIT_ATTN_BACKEND", None)
    visualserver.set_vit_att_backend("auto", defer_validation=True)
    first = visualserver.get_vit_attn_backend()
    q = torch.randn(4, 2, 64, dtype=torch.float16)
    inputs = VarlenAttentionInputs(q, q, q, torch.tensor([0, 4], dtype=torch.int32), 4)
    first.forward(inputs, out=torch.empty_like(q))
    assert calls == [("auto", {"device": q.device, "dtype": q.dtype, "head_dim": 64})]
    assert isinstance(visualserver.get_vit_attn_backend(), SdpaVarlenAttentionBackend)
    first.forward(inputs, out=torch.empty_like(q))
    assert len(calls) == 1


def test_probe_cache_separates_device_dtype_and_layout(monkeypatch):
    from dataclasses import replace

    import torch

    from unit_tests.common.attention.helpers import make_spec

    calls = []
    monkeypatch.setattr(backend_validator, "_validate", lambda *args, **kwargs: calls.append(kwargs) or True)
    backend_validator.validate.cache_clear()
    try:
        spec = make_spec()
        for device, dtype, shape in (
            ("cuda:0", torch.float16, spec),
            ("cuda:1", torch.float16, spec),
            ("cuda:0", torch.bfloat16, spec),
            ("cuda:0", torch.float16, replace(spec, head_size=64)),
            ("cuda:0", torch.float16, spec),
        ):
            assert backend_validator.validate("provider", device=device, dtype=dtype, spec=shape)
        assert len(calls) == 4
    finally:
        backend_validator.validate.cache_clear()

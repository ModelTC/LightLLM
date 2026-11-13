import pytest
import torch

from lightllm.common.attention import selection
from lightllm.common.attention.api import UnsupportedAttention
from lightllm.utils import backend_validator
from lightllm.utils.backend_validator import ProbeResult, ProbeScope


def test_auto_probes_actual_varlen_path_on_assigned_device(monkeypatch):
    calls = []

    def probe(name, **kwargs):
        calls.append((name, kwargs))
        return ProbeResult(name == "sdpa", ProbeScope.VARLEN_NUMERICS, "rejected")

    monkeypatch.setattr(backend_validator, "probe_varlen", probe)
    assert selection.select_varlen_attention_op(device=3) == "sdpa"
    assert calls == [
        (name, {"device": torch.device("cuda:3"), "dtype": None, "head_dim": 80})
        for name in ("fa3", "xformers", "sdpa")
    ]


@pytest.mark.parametrize("requested", ["auto", "fa3", "triton", "missing"])
def test_rejected_ops_are_not_returned(monkeypatch, requested):
    calls = []

    def probe(name, **kwargs):
        calls.append(name)
        return ProbeResult(False, ProbeScope.VARLEN_NUMERICS, "rejected")

    monkeypatch.setattr(backend_validator, "probe_varlen", probe)
    with pytest.raises(selection.BackendSelectionError) as error:
        selection.select_varlen_attention_op(requested)
    expected = list(selection.varlen_attention_ops) if requested == "auto" else [requested]
    assert list(error.value.failures) == expected
    assert calls == ([] if requested == "missing" else expected)


def test_varlen_probe_exercises_selected_op(monkeypatch):
    from lightllm.common.attention.backends.dense.sdpa.varlen import SdpaVarlenAttentionOp

    monkeypatch.setattr(backend_validator.torch.cuda, "is_available", lambda: False)
    assert backend_validator._validate_varlen("sdpa") == (True, None)
    monkeypatch.setattr(SdpaVarlenAttentionOp, "forward", lambda self, inputs, out: out.fill_(float("nan")))
    success, reason = backend_validator._validate_varlen("sdpa")
    assert not success
    assert "mismatch" in reason


@pytest.mark.parametrize("check", ["check_environment", "check_support"])
def test_rejected_probe_does_not_execute_op(monkeypatch, check):
    from lightllm.common.attention.backends.dense.sdpa.varlen import SdpaVarlenAttentionOp

    monkeypatch.setattr(backend_validator.torch.cuda, "is_available", lambda: False)

    def reject(cls, *args):
        raise UnsupportedAttention("layout")

    monkeypatch.setattr(SdpaVarlenAttentionOp, check, classmethod(reject))
    monkeypatch.setattr(SdpaVarlenAttentionOp, "forward", lambda *args, **kwargs: pytest.fail("rejected operator ran"))
    assert backend_validator._validate_varlen("sdpa") == (False, "layout")


def test_worker_initialization_keeps_fixed_forward_and_uses_actual_inputs(monkeypatch):
    import torch

    from lightllm.common.attention.api import VarlenAttentionInputs
    from lightllm.common.attention.backends.dense.sdpa.varlen import SdpaVarlenAttentionOp
    from lightllm.server import visualserver

    calls = []

    def select(requested, **kwargs):
        calls.append((requested, kwargs))
        return "sdpa"

    monkeypatch.setattr(visualserver, "select_varlen_attention_op", select)
    monkeypatch.setattr(visualserver, "VIT_ATTENTION_OP", None)
    visualserver.set_vit_attention_op("auto", defer_validation=True)
    first = visualserver.get_vit_attention_op()
    q = torch.randn(4, 2, 64, dtype=torch.float16)
    inputs = VarlenAttentionInputs(q, q, q, torch.tensor([0, 4], dtype=torch.int32), 4)
    first.forward(inputs, out=torch.empty_like(q))
    assert calls == [("auto", {"device": q.device, "dtype": q.dtype, "head_dim": 64})]
    assert visualserver.get_vit_attention_op() is first
    assert isinstance(first._op, SdpaVarlenAttentionOp)
    assert "forward" not in vars(first)
    first.forward(inputs, out=torch.empty_like(q))
    assert len(calls) == 1


def test_probe_cache_separates_device_dtype_and_layout(monkeypatch):
    from dataclasses import replace

    import torch

    from unit_tests.common.attention.helpers import make_spec

    calls = []
    monkeypatch.setattr(
        backend_validator,
        "_probe",
        lambda *args, **kwargs: calls.append(kwargs) or ProbeResult(True, ProbeScope.PROVIDER_SMOKE),
    )
    backend_validator.probe_backend.cache_clear()
    try:
        spec = make_spec()
        for device, dtype, shape in (
            ("cuda:0", torch.float16, spec),
            ("cuda:1", torch.float16, spec),
            ("cuda:0", torch.bfloat16, spec),
            ("cuda:0", torch.float16, replace(spec, head_size=64)),
            ("cuda:0", torch.float16, spec),
        ):
            assert backend_validator.probe_backend("provider", device=device, dtype=dtype, spec=shape).ok
        assert len(calls) == 4
    finally:
        backend_validator.probe_backend.cache_clear()


def test_empty_device_assignment_does_not_probe(monkeypatch):
    monkeypatch.setattr(backend_validator, "probe_varlen", lambda *a, **k: pytest.fail("empty assignment probed"))
    with pytest.raises(ValueError, match="at least one device"):
        selection.select_varlen_attention_op(devices=())


def test_default_device_is_resolved_before_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(
        selection, "get_device_runtime", lambda device: type("Runtime", (), {"device": torch.device("cpu")})()
    )
    monkeypatch.setattr(
        backend_validator,
        "probe_varlen",
        lambda *a, **kw: calls.append(kw["device"]) or ProbeResult(True, ProbeScope.VARLEN_NUMERICS),
    )
    assert selection.select_varlen_attention_op("sdpa") == "sdpa"
    assert calls == [torch.device("cpu")]

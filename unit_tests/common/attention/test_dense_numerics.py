from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionPhase, CacheLayout, ContextCapacity, UnsupportedAttention
from lightllm.common.attention.api.inputs import DenseInputs, DenseParams
from lightllm.common.attention.backends._workspace import AttentionWorkspace
from lightllm.common.attention.registry import cache_format_to_backend, load_backend_class
from lightllm.common.attention_state import AttentionStateManager
from lightllm.platform import get_device_runtime
from unit_tests.common.attention.helpers import make_batch, make_layer_attention, make_spec, model_stub

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(query, key, value, lengths, q_lengths, window=None):
    rows, start = [], 0
    groups = query.shape[1] // key.shape[1]
    for request, (length, q_length) in enumerate(zip(lengths, q_lengths)):
        k = key[request * 16 : request * 16 + length].float().repeat_interleave(groups, dim=1)
        v = value[request * 16 : request * 16 + length].float().repeat_interleave(groups, dim=1)
        q = query[start : start + q_length].float()
        logits = torch.einsum("qhd,khd->hqk", q, k) * query.shape[-1] ** -0.5
        causal = torch.arange(length, device=q.device)[None, :] <= (
            length - q_length + torch.arange(q_length, device=q.device)[:, None]
        )
        if window is not None:
            causal &= torch.arange(length, device=q.device)[None, :] >= (
                length - q_length + torch.arange(q_length, device=q.device)[:, None] - window[0]
            )
        probs = logits.masked_fill(~causal, -torch.inf).softmax(-1)
        rows.append(torch.einsum("hqk,khd->qhd", probs, v))
        start += q_length
    return torch.cat(rows).to(query.dtype)


@pytest.mark.parametrize("provider", ["triton", "fa3", "flashinfer"])
@pytest.mark.parametrize("phase", list(AttentionPhase))
def test_startup_probe_runs_the_selected_dense_implementation(provider, phase):
    from lightllm.common.attention.probe import probe_dense

    cls = load_backend_class(cache_format_to_backend["None"][provider])
    runtime = get_device_runtime()
    try:
        cls.check_environment(runtime.caps)
    except UnsupportedAttention as exc:
        pytest.skip(str(exc))
    spec = make_spec(
        phase=phase, head_size=64, value_head_size=64, cache=CacheLayout("None", 16, torch.bfloat16, 2, 64, 64)
    )
    success, error = probe_dense(cls, spec, runtime)
    assert success, error


@pytest.mark.parametrize("provider", ["fa3", "flashinfer"])
def test_noncausal_prefill_probe(provider):
    from lightllm.common.attention.probe import probe_dense

    cls = load_backend_class(cache_format_to_backend["None"][provider])
    runtime = get_device_runtime()
    cls.check_environment(runtime.caps)
    spec = make_spec(
        phase=AttentionPhase.PREFILL,
        causal=False,
        head_size=64,
        value_head_size=64,
        cache=CacheLayout("None", 16, torch.bfloat16, 2, 64, 64),
    )
    cls.check_support(spec, runtime.caps)
    success, error = probe_dense(cls, spec, runtime)
    assert success, error


def test_triton_rejects_noncausal_prefill():
    cls = load_backend_class(cache_format_to_backend["None"]["triton"])
    with pytest.raises(UnsupportedAttention, match="non-causal prefill"):
        cls.check_support(make_spec(phase=AttentionPhase.PREFILL, causal=False))

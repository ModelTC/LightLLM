from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionFamily, AttentionPhase, UnsupportedAttention
from lightllm.common.attention.backends.base import BaseAttentionBackend
from unit_tests.common.attention.helpers import make_batch, make_context, make_layer_attention, make_spec, model_stub


def test_graph_copy_validates_all_inputs_before_writing():
    captured = make_batch()
    incoming = make_batch((9, 7))
    incoming = replace(incoming, req_to_token_indexs=captured.req_to_token_indexs, b_seq_len=incoming.b_seq_len.long())
    before = captured.b_req_idx.clone()
    with pytest.raises(ValueError, match="b_seq_len"):
        captured.copy_inputs(incoming)
    torch.testing.assert_close(captured.b_req_idx, before)


def test_graph_copy_keeps_tensor_addresses_and_uses_new_lengths():
    captured = make_batch()
    incoming = make_batch((9, 7), req_to_token_indexs=captured.req_to_token_indexs)
    pointer = captured.b_seq_len.data_ptr()
    captured.copy_inputs(incoming)
    assert captured.b_seq_len.data_ptr() == pointer
    assert captured.b_seq_len.tolist() == [9, 7]
    with pytest.raises(ValueError, match="request table"):
        captured.copy_inputs(make_batch((9, 7)))


@pytest.mark.parametrize("capture", [False, True])
def test_context_capacity_and_backend_identity(capture):
    backend = BaseAttentionBackend(model_stub(), make_spec())
    context = make_context(backend, capture=capture, max_length=16)
    backend.prepare(context, make_batch(req_to_token_indexs=context.batch.req_to_token_indexs))
    with pytest.raises(ValueError, match="capacity"):
        backend.prepare(context, make_batch((17, 2)))
    with pytest.raises(ValueError, match="different attention backend"):
        BaseAttentionBackend(model_stub(), make_spec()).prepare(context, make_batch())
    with pytest.raises(ValueError, match="different attention backend"):
        BaseAttentionBackend(model_stub(), make_spec()).update(context)
    attention = make_layer_attention(BaseAttentionBackend(model_stub(), make_spec()), context, params=None)
    with pytest.raises(ValueError, match="different attention backend"):
        attention.backend.forward(attention, context, inputs=None)


def test_layer_attention_rejects_wrong_phase():
    backend = BaseAttentionBackend(model_stub(), make_spec())
    context = make_context(backend, prepare=True)
    attention = make_layer_attention(backend, context, params=None)
    attention = replace(attention, spec=replace(attention.spec, phase=AttentionPhase.PREFILL))
    with pytest.raises(ValueError, match="phases differ"):
        attention.backend.forward(attention, context, inputs=None)


class _OutputBackend(BaseAttentionBackend):
    def _forward_decode(self, layer, context, inputs, *, out=None):
        return inputs


@pytest.mark.parametrize("strided", [False, True])
def test_output_copy_and_padded_view_preserve_auxiliary_result(strided):
    backend = _OutputBackend(model_stub(), make_spec())
    context = make_context(backend, prepare=True)
    layer = make_layer_attention(backend, context, params=None)
    out = torch.empty(2, 8 if strided else 4, 8)[:, :4]
    gate = torch.ones(2)
    output, actual_gate = layer.backend.forward(layer, context, (torch.ones(2, 32), gate), out=out)
    assert output is out and actual_gate is gate
    assert torch.all(out == 1)
    valid_heads = out[:, :2]
    output, actual_gate = layer.backend.forward(layer, context, (valid_heads, gate), out=out)
    assert output is valid_heads and actual_gate is gate


@pytest.mark.parametrize("invalid", ["dtype", "device", "size", "view_bounds"])
def test_output_rejects_incompatible_buffer(invalid):
    backend = _OutputBackend(model_stub(), make_spec())
    context = make_context(backend, prepare=True)
    layer = make_layer_attention(backend, context, params=None)
    out = torch.empty(2, 4)
    result = torch.ones_like(out)
    if invalid == "dtype":
        result = result.double()
    elif invalid == "device":
        result = result.to("meta")
    elif invalid == "size":
        result = result[:1]
    else:
        result, out = out, out[:1]
    with pytest.raises(ValueError, match="out"):
        layer.backend.forward(layer, context, result, out=out)


def test_spec_configures_backend_before_planning():
    from lightllm.common.attention.api import AttentionExecutionContext, AttentionWorkspace

    model = model_stub(page_size=1, mtp_dynamic_verify=False)
    spec = replace(make_spec(), cache=replace(make_spec().cache, page_size=16), causal=False, dynamic_verify=True)
    backend = BaseAttentionBackend.create(model, spec)
    assert backend.spec is spec
    assert (backend.page_size, backend.causal, backend.dynamic_verify) == (16, False, True)
    context = make_context(backend)
    assert isinstance(context, AttentionExecutionContext)
    assert isinstance(context.workspace, AttentionWorkspace)
    assert context.backend_identity is backend


def test_binding_resolves_context_on_each_forward(monkeypatch):
    backend = _OutputBackend(model_stub(), make_spec())
    first, second = make_context(backend), make_context(backend)
    binding = make_layer_attention(backend, first, params=None)
    contexts = {binding.group_id: first}
    seen = []

    def resolve(layer):
        assert layer is binding
        seen.append(contexts[layer.group_id])
        return contexts[layer.group_id]

    forwarded = []

    def forward(layer, context, inputs, *, out=None):
        assert layer is binding
        forwarded.append(context)
        return inputs

    monkeypatch.setattr(backend, "_forward_decode", forward)
    execution = SimpleNamespace(attention_context=resolve)
    result = torch.ones(2, 32)
    out = torch.empty_like(result)
    assert binding.forward(execution, result, out=out) is out
    contexts[binding.group_id] = second
    assert binding.forward(execution, result, out=out) is out
    assert seen == [first, second]
    assert all(actual is expected for actual, expected in zip(forwarded, (first, second)))
    torch.testing.assert_close(out, result)

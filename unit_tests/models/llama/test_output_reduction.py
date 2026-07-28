from types import SimpleNamespace

import torch

from lightllm.common.basemodel.layer_infer.template import transformer_layer_infer_template
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer


class _IdentityProjection:
    def mm(self, input):
        return input


def test_get_o_reduces_by_default():
    layer = object.__new__(LlamaTransformerLayerInfer)
    layer.tp_o_head_num_ = 1
    layer.head_dim_ = 2
    layer._should_fuse_ar_add_norm = lambda infer_state: True
    reduce_calls = []
    layer._tpsp_reduce = lambda input, infer_state: reduce_calls.append(input) or input

    input = torch.ones(1, 1, 2)
    infer_state = SimpleNamespace(need_dp_prefill_balance=False)
    layer_weight = SimpleNamespace(o_proj=_IdentityProjection())

    output = layer._get_o(input, infer_state, layer_weight)

    assert output.shape == (1, 2)
    assert reduce_calls == [output]


def test_get_o_only_defers_when_requested():
    layer = object.__new__(LlamaTransformerLayerInfer)
    layer.tp_o_head_num_ = 1
    layer.head_dim_ = 2
    reduce_calls = []
    layer._tpsp_reduce = lambda input, infer_state: reduce_calls.append(input) or input

    input = torch.ones(1, 1, 2)
    infer_state = SimpleNamespace(need_dp_prefill_balance=False)
    layer_weight = SimpleNamespace(o_proj=_IdentityProjection())

    output = layer._get_o(input, infer_state, layer_weight, defer_reduction=True)

    assert output.shape == (1, 2)
    assert reduce_calls == []


def test_ineligible_fusion_does_not_allocate_norm_output(monkeypatch):
    layer = object.__new__(LlamaTransformerLayerInfer)
    layer.embed_dim_ = 4

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("norm_out must not be allocated")

    layer.alloc_tensor = fail_allocation
    layer._ffn_norm = lambda input_, _infer_state, _layer_weight: input_.clone()

    reduce_calls = []
    monkeypatch.setattr(
        transformer_layer_infer_template,
        "all_reduce",
        lambda tensor, group: reduce_calls.append((tensor, group)),
    )

    group = SimpleNamespace(flashinfer_reduce=SimpleNamespace(should_use=lambda _tensor: False))
    infer_state = SimpleNamespace(dist_group=group)
    layer_weight = SimpleNamespace(ffn_norm_weight_=SimpleNamespace(weight=torch.ones(4)))
    o = torch.ones((2, 4))
    residual = torch.full((2, 4), 2.0)

    norm_out = layer._reduce_add_ffn_norm(o, residual, infer_state, layer_weight)

    assert len(reduce_calls) == 1
    assert reduce_calls[0][0].data_ptr() == o.data_ptr()
    assert reduce_calls[0][1] is group
    torch.testing.assert_close(residual, torch.full((2, 4), 3.0))
    torch.testing.assert_close(norm_out, residual)

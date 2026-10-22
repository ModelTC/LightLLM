from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek_v4.layer_infer import transformer_layer_infer


@pytest.mark.parametrize("groups", [1, 2, 8])
@pytest.mark.parametrize("tokens", [1, 16, 256])
def test_grouped_o_projection_writes_contiguous_consumer_layout(monkeypatch, groups, tokens):
    torch.manual_seed(1012)
    layer = object.__new__(transformer_layer_infer.DeepseekV4TransformerLayerInfer)
    layer.tp_groups = groups
    layer.qk_rope_head_dim = 2
    layer._select_rope = lambda _: (None, None)
    layer._tpsp_reduce = lambda input, infer_state: input
    monkeypatch.setattr(transformer_layer_infer, "rotary_emb_fwd", lambda *_args, **_kwargs: None)
    allocations = []

    def alloc(shape, dtype, device):
        result = torch.empty(shape, dtype=dtype, device=device)
        allocations.append(result)
        return result

    layer.alloc_tensor = alloc
    x = torch.randn(tokens, groups * 2, 4)
    wa = torch.randn(groups, 8, 3)
    wb = torch.randn(groups * 3, 5)

    def bmm(input, out):
        assert input.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        assert out.untyped_storage().data_ptr() == allocations[0].untyped_storage().data_ptr()
        return torch.bmm(input, wa, out=out)

    def mm(input):
        assert input.is_contiguous()
        assert input.shape == (tokens, groups * 3)
        assert input.untyped_storage().data_ptr() == allocations[0].untyped_storage().data_ptr()
        return input @ wb

    weight = SimpleNamespace(
        o_proj_use_mm=False,
        wo_a_=SimpleNamespace(dim2=3, bmm=bmm),
        wo_b_=SimpleNamespace(mm=mm),
    )
    result = layer._get_o(x, SimpleNamespace(), weight)
    expected = torch.einsum("tgk,gkr->tgr", x.reshape(tokens, groups, 8), wa).reshape(tokens, -1) @ wb
    torch.testing.assert_close(result, expected)
    assert len(allocations) == 1


def test_single_group_o_projection_uses_linear_weight(monkeypatch):
    layer = object.__new__(transformer_layer_infer.DeepseekV4TransformerLayerInfer)
    layer.qk_rope_head_dim = 2
    layer._select_rope = lambda _: (None, None)
    layer._tpsp_reduce = lambda input, infer_state: input
    monkeypatch.setattr(transformer_layer_infer, "rotary_emb_fwd", lambda *_args, **_kwargs: None)
    x = torch.randn(16, 2, 4)
    wa, wb = torch.randn(8, 3), torch.randn(3, 5)
    weight = SimpleNamespace(
        o_proj_use_mm=True,
        wo_a_=SimpleNamespace(mm=lambda input: input @ wa),
        wo_b_=SimpleNamespace(mm=lambda input: input @ wb),
    )
    torch.testing.assert_close(layer._get_o(x, SimpleNamespace(), weight), x.reshape(16, -1) @ wa @ wb)

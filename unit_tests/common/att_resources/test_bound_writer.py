import pytest
import torch

from unit_tests.common.att_resources.test_quantized_full_state import KINDS, make_quant_state
from unit_tests.common.basemodel.attention.test_binding import make_state, bind_test_model
from unit_tests.common.att_resources.test_fp8_e8m0_packed_mla_state import make_state as make_e8m0_state
from lightllm.common.attention_state.full_att import MLAFullAttentionState, MLAConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("kind", (*KINDS, "gqa", "mla", "e8m0"))
def test_writer_retains_native_views_without_state_queries(monkeypatch, kind):
    if kind == "gqa":
        state = make_state(((1,), (3,)), device="cuda")
    elif kind == "mla":
        state = MLAFullAttentionState(MLAConfig(((1,), (3,)), 576, torch.bfloat16), device="cuda")
        state.allocate_storage(8)
    elif kind == "e8m0":
        state = make_e8m0_state(device="cuda")
    else:
        state = make_quant_state(kind, "cuda")
    fields = state.get_transfer_fields()
    for field in fields.values():
        field.tensor.zero_()
    layer = state.config.logical_layer_groups[0][0]
    heads = getattr(state.config, "head_num", 1)
    dim = getattr(state.config, "head_dim", getattr(state.config, "latent_dim", 512))
    if kind in ("gqa", "int4", "int8", "fp8_tensor", "fp8_head"):
        heads *= 2
    torch.manual_seed(38)
    kv = torch.randn(2, heads, dim, device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([2, 5], dtype=torch.int32, device="cuda")
    state.write_full_att_kv(layer, indices, kv)
    expected = {name: field.tensor.clone() for name, field in fields.items()}
    for field in fields.values():
        field.tensor.zero_()
    writer = state.bind_kv_writer(layer)

    def forbidden(*args, **kwargs):
        pytest.fail("bound KV writer consulted State")

    for name in ("get_layer_view", "get_storage_layer_index", "write_full_att_kv", "bind_kv_writer"):
        monkeypatch.setattr(state, name, forbidden)
    writer(token_indices=indices, kv=kv)
    writer(token_indices=indices[:0], kv=kv[:0])
    for name, field in fields.items():
        assert torch.equal(field.tensor.contiguous().view(torch.uint8), expected[name].contiguous().view(torch.uint8))


def test_model_bound_writer_does_not_resolve_physical_layer(monkeypatch):
    state = make_state(((1, 3),), heads=2, dim=128, device="cuda")
    model = bind_test_model([state])
    k, v = model.layers_infer[0].attention.cache

    def forbidden(*args, **kwargs):
        pytest.fail("model writer consulted State after binding")

    monkeypatch.setattr(state, "get_storage_layer_index", forbidden)
    monkeypatch.setattr(state, "get_layer_view", forbidden)
    kv = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([1, 5], dtype=torch.int32, device="cuda")
    model.layers_infer[0].write_full_att_kv(token_indices=ids, kv=kv)
    assert torch.equal(k[ids], kv[:, :2])
    assert torch.equal(v[ids], kv[:, 2:])
    with pytest.raises(ValueError, match="second cache write"):
        model.layers_infer[1].write_full_att_kv(token_indices=ids, kv=kv)

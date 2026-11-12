from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import BMMWeightTpl
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_projection(layer, heads: int, inner: int, cols: int, dtype: torch.dtype):
    projection = BMMWeightTpl.__new__(BMMWeightTpl)
    projection.weight = torch.randn(heads, inner, cols, device="cuda", dtype=dtype) * 0.1
    layer.tp_q_head_num_ = heads
    if inner == 256:
        layer.kv_lora_rank = cols
        layer.layer_num_ = 0
        layer.softmax_scale = inner ** -0.5
        layer.indexer = SimpleNamespace(_get_indices=lambda **kwargs: (None, None))
        state = SimpleNamespace(
            att_state_role="main",
            att_state_manager=Mock(),
            prefill_att_state=SimpleNamespace(prefill_att=lambda **kwargs: kwargs["q"]),
        )
        weight = SimpleNamespace(k_b_proj_=projection)

        def run(x):
            state.get_topk_indices_params = {"hidden_states": None, "q_lora": None}
            return layer._context_attention_kernel(x, None, state, weight)

    else:
        layer.v_head_dim = cols
        layer._tpsp_reduce = lambda input, infer_state: input
        state = SimpleNamespace(is_prefill=False, need_dp_prefill_balance=False)
        weight = SimpleNamespace(v_b_proj_=projection, o_weight_=SimpleNamespace(mm=lambda x: x))

        def run(x):
            return layer._get_o(x, state, weight).view(x.shape[0], heads, cols)

    return projection, run


@pytest.mark.parametrize("rows", [1, 128, 1024, 1659, 4096, 8192])
@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("inner,cols", [(256, 512), (512, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_token_major_projection_matches_reference_and_keeps_guard_rows(rows, heads, inner, cols, dtype):
    torch.manual_seed(272)
    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    projection, run = make_projection(layer, heads, inner, cols, dtype)
    x = torch.randn(rows, heads, inner, device="cuda", dtype=dtype)
    expected = projection.bmm(x.transpose(0, 1)).transpose(0, 1).contiguous()
    slab = torch.full((rows + 2, heads, cols), 113.0, device="cuda", dtype=dtype)
    output = slab[1:-1]
    layer.alloc_tensor = Mock(return_value=output)
    actual = run(x)
    assert actual.data_ptr() == output.data_ptr() and actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert (slab[0] == 113).all() and (slab[-1] == 113).all()


@pytest.mark.parametrize("inner,cols", [(256, 512), (512, 256)])
def test_token_major_projection_accepts_head_padding(inner, cols):
    # TP4 FlashMLA crops a 64-head output to 16 heads without copying it.
    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    projection, run = make_projection(layer, 16, inner, cols, torch.bfloat16)
    x = torch.randn(128, 64, inner, device="cuda", dtype=torch.bfloat16)[:, :16]
    assert not x.is_contiguous()
    expected = projection.bmm(x.transpose(0, 1)).transpose(0, 1)
    actual = run(x)
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("inner,cols", [(256, 512), (512, 256)])
def test_token_major_projection_replays_new_inputs_in_cuda_graph(inner, cols):
    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    projection, run = make_projection(layer, 16, inner, cols, torch.bfloat16)
    x = torch.randn(128, 16, inner, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        run(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(x)
    x.normal_()
    graph.replay()
    expected = projection.bmm(x.transpose(0, 1)).transpose(0, 1)
    assert torch.equal(actual, expected)

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import BMMWeightTpl
from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_layer_and_state() -> tuple[Glm5NextTransformerLayerInfer, SimpleNamespace]:
    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    layer.prefill_token_major_mla_bmm = True
    infer = SimpleNamespace(
        is_prefill=True,
        is_cuda_graph_capture=False,
        need_dp_prefill_balance=False,
        max_cache_len=131072,
        max_kv_seq_len=139264,
    )
    return layer, infer


@pytest.mark.parametrize("rows", [1024, 1659, 4096, 8192])
@pytest.mark.parametrize("inner,cols", [(256, 512), (512, 256)])
def test_token_major_projection_matches_reference_and_keeps_guard_rows(rows: int, inner: int, cols: int) -> None:
    torch.manual_seed(272)
    layer, infer = make_layer_and_state()
    projection = BMMWeightTpl.__new__(BMMWeightTpl)
    projection.weight = torch.randn(64, inner, cols, device="cuda", dtype=torch.bfloat16) * 0.1
    x = torch.randn(rows, 64, inner, device="cuda", dtype=torch.bfloat16)
    expected = projection.bmm(x.transpose(0, 1)).transpose(0, 1).contiguous()
    slab = torch.full((rows + 2, 64, cols), 113.0, device="cuda", dtype=torch.bfloat16)
    output = slab[1:-1]
    layer.alloc_tensor = Mock(return_value=output)
    actual = layer._sparse_mla_bmm(x=x, projection=projection, infer_state=infer, out_features=cols)
    assert actual is output and actual.is_contiguous()
    assert torch.equal(actual, expected)
    assert (slab[0] == 113).all() and (slab[-1] == 113).all()
    layer.alloc_tensor.assert_called_once_with(shape=(rows, 64, cols), dtype=x.dtype, device=x.device)


@pytest.mark.parametrize("case", ["decode", "capture", "balance", "cold", "small", "disabled", "strided"])
def test_unsupported_execution_keeps_reference_bmm(case: str) -> None:
    layer, infer = make_layer_and_state()
    if case == "decode":
        infer.is_prefill = False
    elif case == "capture":
        infer.is_cuda_graph_capture = True
    elif case == "balance":
        infer.need_dp_prefill_balance = True
    elif case == "cold":
        infer.max_cache_len, infer.max_kv_seq_len = 0, 1024
    elif case == "disabled":
        layer.prefill_token_major_mla_bmm = False
    rows = 128 if case == "small" else 1024
    x = torch.randn(rows, 64, 257 if case == "strided" else 256, device="cuda", dtype=torch.bfloat16)[..., :256]
    projection = BMMWeightTpl.__new__(BMMWeightTpl)
    projection.weight = torch.randn(64, 256, 512, device="cuda", dtype=torch.bfloat16)
    reference = projection.bmm(x.transpose(0, 1))
    projection.bmm = Mock(return_value=reference)
    layer.alloc_tensor = Mock(side_effect=AssertionError("unsupported path must not use direct output"))
    actual = layer._sparse_mla_bmm(x=x, projection=projection, infer_state=infer, out_features=512)
    assert torch.equal(actual, reference.transpose(0, 1))
    assert "out" not in projection.bmm.call_args.kwargs
    layer.alloc_tensor.assert_not_called()

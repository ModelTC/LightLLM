from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_quant_kernel import weight_quant
from lightllm.common.quantization.deepgemm import DeepGEMMFP8w8a8B128QuantizationMethod
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.models.glm5_next.layer_infer import transformer_layer_infer as glm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Hopper and DeepGEMM required",
)


def make_weight(values: torch.Tensor, quantized: bool, ue8m0: bool) -> MMWeightTpl:
    owner = MMWeightTpl.__new__(MMWeightTpl)
    owner.bias = None
    if quantized:
        weight, scale = weight_quant(x=values, block_size=128, use_ue8m0_scales=ue8m0)
        owner.mm_param = WeightPack(weight=weight, weight_scale=scale)
        owner.quant_method = DeepGEMMFP8w8a8B128QuantizationMethod()
        owner.quant_method.use_ue8m0_scales = ue8m0
        owner.quant_method.use_packed_ue8m0 = False
    else:
        owner.mm_param = WeightPack(weight=values)
        owner.quant_method = NoQuantization()
    return owner


@pytest.mark.parametrize("rows,intermediate", [(1, 256), (7, 2048), (33, 2048), (256, 2048), (129, 12288)])
@pytest.mark.parametrize("quantized,ue8m0", [(True, True), (True, False), (False, False)])
def test_ffn_matches_main_and_preserves_quantization(
    rows: int, intermediate: int, quantized: bool, ue8m0: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(seed=5512 + rows)
    hidden = 256
    x = torch.randn(size=(rows, hidden), device="cuda", dtype=torch.bfloat16) * 8
    gate_up = make_weight(
        values=torch.randn(size=(2 * intermediate, hidden), device="cuda", dtype=x.dtype) * 0.1,
        quantized=quantized,
        ue8m0=ue8m0,
    )
    down = make_weight(
        values=torch.randn(size=(hidden, intermediate), device="cuda", dtype=x.dtype) * 0.05,
        quantized=quantized,
        ue8m0=ue8m0,
    )
    layer = glm.Glm5NextTransformerLayerInfer.__new__(glm.Glm5NextTransformerLayerInfer)
    layer.embed_dim_, layer.swiglu_limit = hidden, 10.0
    weights = SimpleNamespace(gate_up_proj=gate_up, down_proj=down)
    projected = gate_up.mm(input_tensor=x)
    activated = torch.empty(size=(rows, intermediate), device=x.device, dtype=x.dtype)
    glm.silu_and_mul_fwd(input=projected, output=activated, limit=10.0, alpha=1.0, clamp_up_add_one=False)
    expected = down.mm(input_tensor=activated)
    seen = []
    fused = glm.silu_and_mul_quant_fp8

    def capture(
        input: torch.Tensor, group_size: int, dtype: torch.dtype, limit: float, alpha: float, alloc_func: Callable
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, s = fused(input=input, group_size=group_size, dtype=dtype, limit=limit, alpha=alpha, alloc_func=alloc_func)
        seen.append((q.clone(), s.clone()))
        return q, s

    monkeypatch.setattr(target=glm, name="silu_and_mul_quant_fp8", value=capture)
    actual = layer._ffn_tp(input=x, infer_state=None, layer_weight=weights)
    torch.testing.assert_close(actual=actual, expected=expected, atol=0, rtol=0)
    assert len(seen) == int(quantized and ue8m0)
    if seen:
        q, s = per_token_group_quant_fp8(
            x=activated, group_size=128, column_major_scales=True, scale_tma_aligned=True, use_ue8m0_scales=True
        )
        torch.testing.assert_close(actual=seen[0][0].view(torch.uint8), expected=q.view(torch.uint8), atol=0, rtol=0)
        torch.testing.assert_close(actual=seen[0][1], expected=s, atol=0, rtol=0)


def test_ffn_cuda_graph_reads_changed_inputs() -> None:
    torch.manual_seed(seed=5513)
    hidden, intermediate, rows = 256, 2048, 7
    x = torch.randn(size=(rows, hidden), device="cuda", dtype=torch.bfloat16)
    layer = glm.Glm5NextTransformerLayerInfer.__new__(glm.Glm5NextTransformerLayerInfer)
    layer.embed_dim_, layer.swiglu_limit = hidden, 10.0
    weights = SimpleNamespace(
        gate_up_proj=make_weight(
            values=torch.randn(size=(2 * intermediate, hidden), device="cuda", dtype=x.dtype) * 0.1,
            quantized=True,
            ue8m0=True,
        ),
        down_proj=make_weight(
            values=torch.randn(size=(hidden, intermediate), device="cuda", dtype=x.dtype) * 0.05,
            quantized=True,
            ue8m0=True,
        ),
    )
    layer._ffn_tp(input=x, infer_state=None, layer_weight=weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph=graph):
        actual = layer._ffn_tp(input=x, infer_state=None, layer_weight=weights)
    for factor in (1.0, 12.0):
        x.normal_().mul_(factor)
        projected = weights.gate_up_proj.mm(input_tensor=x)
        activated = torch.empty(size=(rows, intermediate), device=x.device, dtype=x.dtype)
        glm.silu_and_mul_fwd(input=projected, output=activated, limit=10.0, alpha=1.0, clamp_up_add_one=False)
        expected = weights.down_proj.mm(input_tensor=activated)
        graph.replay()
        torch.testing.assert_close(actual=actual, expected=expected, atol=0, rtol=0)

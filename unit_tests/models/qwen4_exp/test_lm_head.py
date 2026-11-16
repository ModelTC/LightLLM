import pytest
import torch

from lightllm.models.qwen4_exp.layer_weights.lm_head_weight import Qwen4ExpLMHeadWeight

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 3, 8, 9])
def test_head_shape_precision_and_explicit_output(dtype, rows):
    torch.manual_seed(3107)
    head = Qwen4ExpLMHeadWeight(128, 257, "head.weight", dtype)
    head.weight.normal_(std=0.02)
    hidden = torch.randn(rows, 128, device="cuda", dtype=dtype)
    result = head(hidden.T)
    assert result.shape == (257, rows) and result.dtype == dtype and result.is_contiguous()
    expected = (head.weight.float() @ hidden.float().T).to(dtype)
    tolerance = max(1e-6, torch.finfo(dtype).eps)
    torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)
    out = torch.empty(257, rows * 2, device="cuda", dtype=dtype)[:, ::2]
    assert head(hidden.T, out=out) is out
    torch.testing.assert_close(out, result, atol=0, rtol=0)


def test_trained_geometry_batch_invariance_and_graph_replay():
    torch.manual_seed(3107)
    head = Qwen4ExpLMHeadWeight(2560, 62080, "head.weight", torch.bfloat16)
    head.weight.normal_(std=0.02)
    hidden = torch.randn(24, 2560, device="cuda", dtype=torch.bfloat16)
    singles = torch.cat([head(row[:, None]) for row in hidden], dim=1)
    for chunk in (2, 3, 5, 7, 8, 9, 15, 24):
        result = torch.cat([head(hidden[start : start + chunk].T) for start in range(0, len(hidden), chunk)], dim=1)
        torch.testing.assert_close(result, singles, atol=0, rtol=0)

    inputs = hidden[:3].clone()
    for _ in range(3):
        head(inputs.T)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = head(inputs.T)
    inputs.add_(0.125)
    graph.replay()
    torch.testing.assert_close(captured, head(inputs.T), atol=0, rtol=0)
    graph.reset()

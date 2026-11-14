import pytest
import torch

from lightllm.models.qwen4_exp.position import combine_mrope


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [0, 1, 32, 512])
@pytest.mark.parametrize("axes", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rotary_axis_selection_matches_cpu_on_graph_replay(rows, axes, dtype):
    shape = (3, rows, 66) if axes else (rows, 66)
    cos = torch.randn(shape, device="cuda", dtype=dtype)[..., ::2]
    sin = torch.randn(shape, device="cuda", dtype=dtype)[..., ::2]
    sections = [8, 8, 0]
    combine_mrope(cos, sin, sections)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = combine_mrope(cos, sin, sections)
    for _ in range(3):
        cos.add_(1)
        sin.mul_(2)
        graph.replay()
        expected = combine_mrope(cos.cpu(), sin.cpu(), sections)
        for result, reference in zip(actual, expected):
            assert result.is_contiguous()
            torch.testing.assert_close(result.cpu(), reference, atol=0, rtol=0)
    graph.reset()

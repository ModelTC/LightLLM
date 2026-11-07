import pytest
import torch

from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.common.quantization.aligned_projection import OutputAlignedProjection

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(autouse=True)
def current_device(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_unaligned_projection_preserves_loading_layout_outputs_and_graph_updates(dtype):
    torch.manual_seed(3826)
    dimensions = [256, 256, 768, 768, 6, 6]
    method = OutputAlignedProjection()
    pack, parts = method.create_weight(dimensions, 2560, dtype, 0)
    weights = [torch.randn(width, 2560, device="cuda", dtype=dtype) * 0.02 for width in dimensions]
    for source, part in zip(weights, parts):
        method.load_weight(source, part)
    assert pack.weight.shape == (2060, 2560)
    assert torch.equal(pack.weight, torch.cat(weights))
    assert method._padded_weight.shape == (2064, 2560)
    assert not torch.count_nonzero(method._padded_weight[2060:])

    hidden = torch.randn(24, 2560, device="cuda", dtype=dtype)

    def project(value, out=None):
        return method.apply(value, pack, out=out, use_custom_tensor_mananger=False)

    singles = torch.cat([project(row[None]) for row in hidden])
    for count in (0, 1, 2, 3, 8, 9, 24):
        actual = project(hidden[:count])
        assert actual.shape == (count, 2060) and actual.is_contiguous()
        torch.testing.assert_close(actual, singles[:count], atol=0, rtol=0)
        out = torch.empty(count, 4120, device="cuda", dtype=dtype)[:, ::2]
        assert project(hidden[:count], out) is out
        torch.testing.assert_close(out, actual, atol=0, rtol=0)

    inputs = hidden[:3].clone()
    for _ in range(3):
        project(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = project(inputs)
    inputs.add_(0.125)
    graph.replay()
    torch.testing.assert_close(captured, project(inputs), atol=0, rtol=0)
    method.load_weight(weights[-1] + 0.25, parts[-1])
    graph.replay()
    torch.testing.assert_close(captured, project(inputs), atol=0, rtol=0)
    assert not torch.count_nonzero(method._padded_weight[2060:])
    graph.reset()


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 2064), (torch.float32, 2060)])
def test_aligned_and_fp32_paths_keep_native_projection(dtype, width):
    method = OutputAlignedProjection()
    pack, parts = method.create_weight([width], 128, dtype, 0)
    source = torch.randn(width, 128, device="cuda", dtype=dtype)
    method.load_weight(source, parts[0])
    assert method._padded_weight is None
    hidden = torch.randn(3, 128, device="cuda", dtype=dtype)
    result = method.apply(hidden, pack, use_custom_tensor_mananger=False)
    torch.testing.assert_close(result, hidden @ source.T, atol=0, rtol=0)


def test_cached_outputs_and_graph_replay_keep_live_buffers_independent():
    method = OutputAlignedProjection()
    pack, parts = method.create_weight([2060], 2560, torch.bfloat16, 0)
    source = torch.randn(2060, 2560, device="cuda", dtype=torch.bfloat16) * 0.02
    method.load_weight(source, parts[0])
    inputs = torch.randn(3, 2560, device="cuda", dtype=torch.bfloat16)
    expected = method.apply(inputs, pack, use_custom_tensor_mananger=False)
    graph = torch.cuda.CUDAGraph()
    g_cache_manager.cache_env_in()
    try:
        live = method.apply(inputs, pack)
        for _ in range(4):
            other = method.apply(inputs + 0.25, pack)
            assert other.data_ptr() != live.data_ptr()
            torch.testing.assert_close(live, expected, atol=0, rtol=0)
            del other
        with torch.cuda.graph(graph):
            captured = method.apply(inputs, pack)
        inputs.add_(0.125)
        graph.replay()
        torch.testing.assert_close(
            captured, method.apply(inputs, pack, use_custom_tensor_mananger=False), atol=0, rtol=0
        )
        torch.testing.assert_close(live, expected, atol=0, rtol=0)
    finally:
        graph.reset()
        g_cache_manager.cache_env_out()

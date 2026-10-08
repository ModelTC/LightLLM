from types import SimpleNamespace

import pytest
import torch

from lightllm.common.layers.rope import get_rope


@pytest.mark.parametrize("rope_type", ["default", "mrope", "dynamic", "yarn", "su", "llama3"])
def test_cache_builders_use_requested_device(rope_type):
    config = {
        "max_position_embeddings": 16,
        "original_max_position_embeddings": 8,
        "rope_scaling": {
            "rope_type": rope_type,
            "factor": 2.0,
            "original_max_position_embeddings": 8,
            "short_factor": [1.0] * 16,
            "long_factor": [2.0] * 16,
        },
    }
    rope = get_rope(config, 32, 20, torch.bfloat16, device=torch.device("cpu"))
    # Non-contiguous, multi-axis positions must retain their layout and repeats.
    positions = torch.tensor([[0, 1, 7, 0], [8, 9, 15, 8], [19, 3, 2, 19]])[:, ::2]
    cos, sin = rope.get_cos_sin(positions)
    assert rope.cos_cached.device.type == rope.sin_cached.device.type == "cpu"
    assert cos.dtype == sin.dtype == torch.bfloat16
    torch.testing.assert_close(cos, rope.cos_cached[positions], rtol=0, atol=0)
    torch.testing.assert_close(sin, rope.sin_cached[positions], rtol=0, atol=0)


def _rotate_reference(x, cos, sin):
    half = cos.shape[-1]
    result = x.clone()
    x0, x1 = x[..., :half].float(), x[..., half : 2 * half].float()
    result[..., :half] = x0 * cos[:, None].float() - x1 * sin[:, None].float()
    result[..., half : 2 * half] = x0 * sin[:, None].float() + x1 * cos[:, None].float()
    return result


def test_custom_rotary_implementation():
    def torch_rotary(*, q, k, cos, sin, partial_rotary_factor):
        assert partial_rotary_factor == 0.5
        q.copy_(_rotate_reference(q, cos, sin))
        k.copy_(_rotate_reference(k, cos, sin))

    rope = get_rope(
        {"max_position_embeddings": 16, "partial_rotary_factor": 0.5},
        32,
        16,
        torch.float32,
        torch.device("cpu"),
        rotary_impl=torch_rotary,
    )
    cos, sin = rope.get_cos_sin(torch.tensor([0, 7]))
    q, k = torch.randn(2, 4, 32), torch.randn(2, 2, 32)
    expected_q, expected_k = _rotate_reference(q, cos, sin), _rotate_reference(k, cos, sin)
    rope(q, k, cos, sin)
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(k, expected_k, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for rotary kernels")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("partial_factor", [0.25, 1.0])
@pytest.mark.parametrize("has_k", [True, False])
def test_rotary_component_matches_reference(dtype, partial_factor, has_k):
    config = {"max_position_embeddings": 32, "partial_rotary_factor": partial_factor}
    rope = get_rope(config, 64, 32, dtype, device=torch.device("cuda"))
    positions = torch.tensor([0, 1, 7, 19, 31], device="cuda")
    cos, sin = rope.get_cos_sin(positions)
    # Strided Q and a K view into a shared KV allocation exercise the layer call contract.
    q = torch.randn(5, 8, 64, dtype=dtype, device="cuda")[:, ::2]
    kv = torch.randn(5, 4, 64, dtype=dtype, device="cuda")
    before_kv = kv.clone()
    k = kv[:, :2] if has_k else None
    expected_q = _rotate_reference(q, cos, sin)
    expected_k = _rotate_reference(k, cos, sin) if has_k else None
    q_ptr = q.data_ptr()
    rope(q, k, cos, sin)
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert q.data_ptr() == q_ptr
    torch.testing.assert_close(q, expected_q, rtol=tolerance, atol=tolerance)
    rotary_dim = int(64 * partial_factor)
    torch.testing.assert_close(q[..., rotary_dim:], expected_q[..., rotary_dim:], rtol=0, atol=0)
    if has_k:
        torch.testing.assert_close(k, expected_k, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(kv[:, 2:], before_kv[:, 2:], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for MRoPE")
@pytest.mark.parametrize("interleaved", [False, True])
def test_mrope_layout_is_independent_of_frequency_scaling(interleaved):
    config = {
        "max_position_embeddings": 32,
        "partial_rotary_factor": 0.5,
        "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 16},
    }
    rope = get_rope(
        config,
        32,
        32,
        torch.float32,
        torch.device("cuda"),
        mrope_section=[3, 3, 2],
        mrope_interleaved=interleaved,
    )
    positions = torch.tensor([[0, 1, 7, 15, 31], [3, 4, 8, 11, 21], [9, 6, 2, 19, 30]], device="cuda")
    cos, sin = rope.get_cos_sin(positions)
    channels = torch.arange(8, device="cuda")
    axes = channels % 3 if interleaved else torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], device="cuda")
    selected_cos = cos[axes, :, channels].T
    selected_sin = sin[axes, :, channels].T
    q = torch.randn(5, 4, 32, device="cuda")
    k = torch.randn(5, 2, 32, device="cuda")
    expected_q = _rotate_reference(q, selected_cos, selected_sin)
    expected_k = _rotate_reference(k, selected_cos, selected_sin)
    rope(q, k, cos, sin)
    torch.testing.assert_close(q, expected_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k, expected_k, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_rotary_cuda_graph_reuses_position_buffers():
    rope = get_rope(
        {"max_position_embeddings": 32, "partial_rotary_factor": 0.5},
        64,
        32,
        torch.float32,
        torch.device("cuda"),
    )
    q_input = torch.randn(3, 4, 64, device="cuda")
    k_input = torch.randn(3, 2, 64, device="cuda")
    q, k = q_input.clone(), k_input.clone()
    positions = torch.tensor([0, 7, 15], device="cuda")
    cos, sin = rope.get_cos_sin(positions)
    rope(q, k, cos, sin)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rope(q, k, cos, sin)
    for offset in (0, 1, 3):
        next_cos, next_sin = rope.get_cos_sin(positions + offset)
        cos.copy_(next_cos)
        sin.copy_(next_sin)
        q.copy_(q_input)
        k.copy_(k_input)
        graph.replay()
        torch.testing.assert_close(q, _rotate_reference(q_input, next_cos, next_sin), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k, _rotate_reference(k_input, next_cos, next_sin), rtol=1e-5, atol=1e-5)


def test_mtp_shares_rope_component():
    from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel

    rope = get_rope({"max_position_embeddings": 16}, 32, 16, torch.float32, torch.device("cpu"))
    draft = Qwen3_5MTPModel.__new__(Qwen3_5MTPModel)
    draft.main_model = SimpleNamespace(rope=rope)
    draft._init_custom()
    assert draft.rope is rope

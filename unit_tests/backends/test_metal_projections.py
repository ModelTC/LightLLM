"""GPU checks for the native projection specializations and their fallbacks."""

import pytest
import torch

from lightllm.backends.metal.gemm import project as project_matrix
from lightllm.backends.metal.gemv import project as project_vector


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"),
    reason="Requires MPS and torch.mps.compile_shader",
)


@pytest.mark.parametrize("batch,rows", [(4, 4096), (7, 4097), (8, 5120), (16, 7168), (16, 8224), (16, 16384)])
def test_small_batch_matrix_matches_cpu(batch, rows):
    generator = torch.Generator().manual_seed(98012)
    host_x = torch.randn(batch, 1024, dtype=torch.float16, generator=generator)
    host_w = torch.randn(rows, 1024, dtype=torch.float16, generator=generator) * 0.02
    expected = host_x.float() @ host_w.float().t()
    # Contiguous views may still have a nonzero storage offset.
    x_storage = torch.empty(batch + 1, 1024, dtype=torch.float16, device="mps")
    w_storage = torch.empty(rows + 1, 1024, dtype=torch.float16, device="mps")
    x, weight = x_storage[1:], w_storage[1:]
    x.copy_(host_x)
    weight.copy_(host_w)
    output_storage = torch.empty(batch + 1, rows, dtype=x.dtype, device=x.device)
    out = output_storage[1:]
    with torch.inference_mode():
        result = project_matrix(x, weight, out=out)
    assert result is out
    torch.testing.assert_close(result.cpu().float(), expected, atol=0.002, rtol=0.003)
    torch.testing.assert_close(x.cpu(), host_x, atol=0, rtol=0)
    torch.testing.assert_close(weight.cpu(), host_w, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_vector_projection_bias_strides_and_offsets(dtype):
    generator = torch.Generator().manual_seed(535)
    size, rows = 1031, 1027
    host_x = torch.randn(1, size, dtype=dtype, generator=generator)
    host_w = torch.randn(rows, size, dtype=dtype, generator=generator) * 0.02
    host_bias = torch.randn(rows, dtype=dtype, generator=generator) * 0.1
    expected = host_x.float() @ host_w.float().t() + host_bias.float()
    x_storage = torch.empty(1, 2 * size + 1, dtype=dtype, device="mps")
    w_storage = torch.empty(rows, 2 * size + 1, dtype=dtype, device="mps")
    out_storage = torch.empty(1, 2 * rows + 1, dtype=dtype, device="mps")
    x, weight, out = x_storage[:, 1::2], w_storage[:, 1::2], out_storage[:, 1::2]
    x.copy_(host_x)
    weight.copy_(host_w)
    with torch.inference_mode():
        result = project_vector(x, weight, out=out, bias=host_bias.to("mps"))
    assert result is out
    torch.testing.assert_close(
        result.cpu().float(),
        expected,
        atol=0.002 if dtype == torch.float16 else 1e-5,
        rtol=0.003 if dtype == torch.float16 else 1e-5,
    )


def test_unsupported_matrix_shapes_and_aliases_select_fallback():
    x = torch.randn(16, 1024, dtype=torch.float16, device="mps")
    weight = torch.randn(7168, 1024, dtype=x.dtype, device=x.device)
    assert project_matrix(x[:1], weight) is None
    assert project_matrix(x[:, ::2], weight[:, ::2]) is None
    assert project_matrix(x.float(), weight.float()) is None
    assert project_matrix(x, weight, bias=weight[:, 0]) is None
    assert project_matrix(x, weight, out=x) is None
    # A different shape can still alias the same storage as the input.
    storage = torch.empty(16 * 7168, dtype=x.dtype, device=x.device)
    assert project_matrix(storage[: 16 * 1024].view(16, 1024), weight, out=storage.view(16, 7168)) is None


@pytest.mark.parametrize("size,rows", [(1024, 7168), (1024, 8224), (1024, 5120), (3584, 1024), (2048, 1027)])
def test_vector_load_projection_aligned_offsets_and_bias(size, rows):
    generator = torch.Generator().manual_seed(81703)
    host_x = torch.randn(1, size, dtype=torch.float16, generator=generator)
    host_w = torch.randn(rows, size, dtype=torch.float16, generator=generator) * 0.02
    bias = torch.randn(rows, dtype=torch.float16, generator=generator) * 0.1
    expected = host_x.float() @ host_w.float().t() + bias.float()
    x_storage = torch.empty(2, size, dtype=torch.float16, device="mps")
    w_storage = torch.empty(rows + 1, size, dtype=torch.float16, device="mps")
    x, weight = x_storage[1:], w_storage[1:]
    x.copy_(host_x)
    weight.copy_(host_w)
    out_storage = torch.empty(1, rows * 2 + 1, dtype=torch.float16, device="mps")
    out = out_storage[:, 1::2]
    actual = project_vector(x, weight, out=out, bias=bias.to("mps"))
    assert actual is out
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0.002, rtol=0.003)

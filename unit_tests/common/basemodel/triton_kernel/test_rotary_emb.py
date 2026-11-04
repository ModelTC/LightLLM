import pytest
import torch

from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim,partial", [(64, 1.0), (128, 1.0), (128, 0.5)])
@pytest.mark.parametrize("with_k", [False, True])
def test_rotary_preserves_product_rounding_and_graph_updates(dtype, head_dim, partial, with_k):
    torch.manual_seed(124)
    q = torch.randn(17, 12, head_dim, dtype=dtype, device="cuda")[:, ::2]
    k = torch.randn(17, 4, head_dim, dtype=dtype, device="cuda")[:, ::2] if with_k else None
    angle = torch.randn(17, int(head_dim * partial) // 2, device="cuda")
    cos, sin = angle.cos().to(dtype), angle.sin().to(dtype)
    original_q = q.clone()
    original_k = k.clone() if k is not None else None

    def expected(value):
        rotated_dim = int(head_dim * partial)
        left, right = value[..., :rotated_dim].chunk(2, -1)
        c, s = cos[:, None], sin[:, None]
        result = value.clone()
        result[..., :rotated_dim] = torch.cat((left * c - right * s, right * c + left * s), -1)
        return result

    target_q = expected(original_q)
    target_k = expected(original_k) if with_k else None
    # These inputs distinguish separate product rounding from FP32 fusion.
    rotary_emb_fwd(q, k, cos, sin, partial)
    assert not torch.equal(q, target_q)
    q.copy_(original_q)
    if with_k:
        k.copy_(original_k)
    rotary_emb_fwd(q, k, cos, sin, partial, round_intermediates=True)
    torch.testing.assert_close(q, target_q, rtol=0, atol=0)
    if with_k:
        torch.testing.assert_close(k, target_k, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            rotary_emb_fwd(q, k, cos, sin, partial, round_intermediates=True)
        q.copy_(original_q * 0.75)
        if with_k:
            k.copy_(original_k * -0.5)
        cos.copy_((angle + 0.3).cos())
        sin.copy_((angle + 0.3).sin())
        target_q = expected(q)
        target_k = expected(k) if with_k else None
        graph.replay()
        torch.testing.assert_close(q, target_q, rtol=0, atol=0)
        if with_k:
            torch.testing.assert_close(k, target_k, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        graph.reset()

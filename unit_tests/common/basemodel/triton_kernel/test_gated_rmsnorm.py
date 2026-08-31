import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.gated_rmsnorm import (
    gated_rmsnorm_forward,
)


@pytest.mark.parametrize("norm_before_gate", [True, False])
def test_gated_rmsnorm_accepts_strided_3d_gate(norm_before_gate):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for gated RMSNorm test")

    torch.manual_seed(123)
    tokens, heads, head_dim = 5, 4, 128
    packed_dim = heads * head_dim + 256
    x = torch.randn((tokens * heads, head_dim), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((head_dim,), device="cuda", dtype=torch.bfloat16)
    packed = torch.randn((tokens, packed_dim), device="cuda", dtype=torch.bfloat16)
    z = packed[:, 128 : 128 + heads * head_dim].view(tokens, heads, head_dim)
    assert not z.is_contiguous()

    run_config = {"BLOCK_N": head_dim, "num_warps": 1}
    expected = gated_rmsnorm_forward(
        x=x,
        weight=weight,
        bias=None,
        eps=1e-6,
        z=z.contiguous().view(-1, head_dim),
        norm_before_gate=norm_before_gate,
        run_config=run_config,
    )
    actual = gated_rmsnorm_forward(
        x=x,
        weight=weight,
        bias=None,
        eps=1e-6,
        z=z,
        norm_before_gate=norm_before_gate,
        run_config=run_config,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

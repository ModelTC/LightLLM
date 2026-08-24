import pytest
import torch

from lightllm.models.qwen3next.triton_kernel.shared_expert_gate import sigmoid_mul_


@pytest.mark.parametrize("gate_width", [1, 6144])
def test_sigmoid_mul_matches_unfused_dtype_rounding(gate_width):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(2026)
    x = torch.randn((17, 6144), device="cuda", dtype=torch.bfloat16) * 2
    gate = torch.randn((17, gate_width), device="cuda", dtype=torch.bfloat16) * 3

    expected = x.clone()
    expected.mul_(gate.clone().sigmoid_())

    actual = x.clone()
    sigmoid_mul_(actual, gate)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

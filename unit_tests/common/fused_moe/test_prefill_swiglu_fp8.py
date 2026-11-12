import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import (
    silu_and_mul_fwd,
    silu_and_mul_quant_fp8,
)
from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 7, 896, 4096, 8192, 23168, 32768])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("row_gap", [0, 16])
def test_explicit_alpha_preserves_activation_rounding_and_ue8m0(rows: int, dtype: torch.dtype, row_gap: int) -> None:
    torch.manual_seed(358)
    intermediate = 2048
    x = (torch.randn(rows, 2 * intermediate + row_gap, device="cuda", dtype=dtype) * 6)[:, : 2 * intermediate]
    # Zero and tiny groups exercise the scale floor as well as clamped values.
    x[0, :128] = 0
    x[0, 128:256] = 1e-6
    activated = torch.empty(rows, intermediate, device="cuda", dtype=dtype)
    silu_and_mul_fwd(x, activated, alpha=1.0, limit=10.0, clamp_up_add_one=False)
    expected_q, expected_scale = per_token_group_quant_fp8(
        activated, 128, column_major_scales=True, scale_tma_aligned=True, use_ue8m0_scales=True
    )
    actual_q, actual_scale = silu_and_mul_quant_fp8(
        x, 128, torch.float8_e4m3fn, limit=10.0, alpha=1.0, run_config={"GROUPS_PER_CTA": 4, "num_warps": 1}
    )
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_scale, expected_scale)
    assert actual_scale.stride() == expected_scale.stride()
    assert torch.equal(torch.log2(actual_scale), torch.log2(actual_scale).round())


@pytest.mark.parametrize("rows", [1, 128, 2048])
def test_default_activation_path_still_matches_unfused(rows: int) -> None:
    torch.manual_seed(1679)
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    activated = torch.empty(rows, 2048, device="cuda", dtype=x.dtype)
    silu_and_mul_fwd(x, activated)
    expected_q, expected_scale = per_token_group_quant_fp8(
        activated, 128, column_major_scales=True, scale_tma_aligned=True, use_ue8m0_scales=True
    )
    actual_q, actual_scale = silu_and_mul_quant_fp8(x, 128, torch.float8_e4m3fn)
    assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(actual_scale, expected_scale)

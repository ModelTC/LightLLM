import pytest
import torch

import lightllm.common.basemodel
from lightllm.common.quantization.awq_marlin import can_prepare_awq, dense_awq_marlin, prepare_awq
from lightllm.common.quantization.quantize_method import WeightPack


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("widths,tokens", [((192, 64, 64), 17), ((512, 256, 256), 128), ((512, 256, 256), 129)])
def test_marlin_repacking_merged_awq_projection(dtype, widths, tokens):
    torch.manual_seed(12)
    height = 256
    width = sum(widths)
    packed = torch.randint(-(2 ** 31), 2 ** 31 - 1, (height, width // 8), dtype=torch.int32, device="cuda")
    zeros = torch.randint_like(packed[: height // 128], -(2 ** 31), 2 ** 31 - 1)
    scales = torch.rand(height // 128, width, dtype=dtype, device="cuda") / 64
    pack = WeightPack(packed, scales, zeros)
    if not can_prepare_awq(pack, "dense"):
        pytest.skip("Standalone Marlin modules for Hopper required")
    shifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device="cuda", dtype=torch.int32) * 4
    values = ((packed[..., None] >> shifts) & 15).view(height // 128, 128, width)
    decoded_zeros = ((zeros[..., None] >> shifts) & 15).view(height // 128, width)
    reference_weight = ((values - decoded_zeros[:, None]).float() * scales.float()[:, None]).view(height, width)
    activations = torch.randn(tokens, height, device="cuda", dtype=dtype)
    expected = activations.float() @ reference_weight
    original_ptr = packed.data_ptr()
    prepare_awq(pack)
    assert pack.weight.data_ptr() == original_ptr
    out = torch.empty(tokens, width, dtype=dtype, device="cuda")
    assert dense_awq_marlin(activations, pack, out) is out
    relative_error = (out.float() - expected).norm() / expected.norm()
    assert relative_error < 0.008

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
from lightllm.models.glm5_next_mtp.triton_kernel.fused_eh_norm import fused_mtp_norm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 4, 16, 64, 512, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale", [1e-4, 1.0, 100.0])
def test_fused_mtp_norm_keeps_intermediate_rounding_and_inplace_updates(rows, dtype, scale):
    torch.manual_seed(5304)
    h = 4096
    # A row gap also exercises the strides used by projected hidden-state views.
    e = (torch.randn(rows, h + 16, dtype=dtype, device="cuda") * scale)[:, :h]
    p = (torch.randn(rows, h + 16, dtype=dtype, device="cuda") * scale)[:, :h]
    er, pr = e.clone(), p.clone()
    wm, we, wh = [torch.randn(h, dtype=dtype, device="cuda") * 0.2 + 1 for _ in range(3)]
    rmsnorm_forward(pr, wm, 1e-6, out=pr)
    rmsnorm_forward(er, we, 1e-6, out=er)
    rmsnorm_forward(pr, wh, 1e-6, out=pr)
    expected = torch.cat((er, pr), dim=-1)
    actual = fused_mtp_norm(e, p, wm, we, wh, 1e-6)
    assert torch.equal(actual, expected)
    assert torch.equal(e, er)
    assert torch.equal(p, pr)

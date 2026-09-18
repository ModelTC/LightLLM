import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.fused_gdn_gating import fused_gdn_gating

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


def test_prefill_beta_keeps_fp32_sigmoid():
    b = torch.tensor([[0.5]], device="cuda", dtype=torch.bfloat16)
    a = torch.zeros_like(b)
    A_log = torch.zeros(1, device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros_like(A_log)

    _, beta = fused_gdn_gating(A_log, a, b, dt_bias, run_config={"BLK_HEADS": 8, "num_warps": 1})

    assert beta.dtype == torch.float32
    torch.testing.assert_close(beta, torch.sigmoid(b.float()), rtol=0, atol=1e-7)

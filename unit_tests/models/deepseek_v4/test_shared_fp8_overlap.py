import unittest
import torch


def _hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


class TestSharedFp8AndOverlap(unittest.TestCase):
    @unittest.skipUnless(_hopper(), "requires SM90+ CUDA FP8")
    def test_shared_ue8m0_quant_matches_parent(self):
        import lightllm.common.basemodel  # initialize dependencies before the DS4 kernel
        from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8
        from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
        from lightllm.models.deepseek_v4.triton_kernel.silu_mul_fp8 import fused_silu_mul_quant_fp8

        torch.manual_seed(17)
        x = torch.randn((3, 4096), device="cuda", dtype=torch.bfloat16).contiguous()
        x[:, :4] = torch.tensor([-11.0, -10.0, 10.0, 11.0], device="cuda", dtype=torch.bfloat16)
        q, scale = fused_silu_mul_quant_fp8(
            x, lambda shape, dtype, device: torch.empty(shape, dtype=dtype, device=device)
        )
        activated = torch.empty((3, 2048), device="cuda", dtype=torch.bfloat16)
        silu_and_mul_fwd(x, activated, limit=10.0)
        expected_q, expected_scale = per_token_group_quant_fp8(
            activated,
            128,
            dtype=torch.float8_e4m3fn,
            column_major_scales=True,
            scale_tma_aligned=True,
            use_ue8m0_scales=True,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(q, expected_q))
        self.assertTrue(torch.equal(scale, expected_scale))
        self.assertEqual(scale.stride(), (1, 4))

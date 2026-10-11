import unittest

import torch


def _cuda_hopper():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _input(rows, kind):
    if kind == "boundary":
        pattern = torch.tensor([-11.0, -10.0, -0.125, 0.125, 10.0, 11.0], device="cuda", dtype=torch.bfloat16)
        gate = pattern.repeat((2048 + pattern.numel() - 1) // pattern.numel())[:2048]
        up = pattern.flip(0).repeat((2048 + pattern.numel() - 1) // pattern.numel())[:2048]
        return torch.cat((gate, up)).repeat(rows, 1).contiguous()
    if kind == "zero_tiny":
        x = torch.zeros((rows, 4096), device="cuda", dtype=torch.bfloat16)
        x[:, 0:4] = torch.tensor([1e-5, -1e-5, 1e-3, -1e-3], device="cuda", dtype=torch.bfloat16)
        return x
    if kind == "random":
        torch.manual_seed(101 + rows)
        x = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)
        x[:, 0:4] = torch.tensor([-11.0, -10.0, 10.0, 11.0], device="cuda", dtype=torch.bfloat16)
        return x
    raise AssertionError(kind)


def _parent_activation(x):
    from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd

    out = torch.empty((x.shape[0], 2048), dtype=torch.bfloat16, device=x.device)
    silu_and_mul_fwd(x, out, limit=10.0)
    return out


def _parent_quant(x, *, ue8m0):
    from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8

    return per_token_group_quant_fp8(
        x, 128, dtype=torch.float8_e4m3fn, column_major_scales=True, scale_tma_aligned=True, use_ue8m0_scales=ue8m0
    )


def _assert_match(q, scale, parent_q, parent_scale):
    torch.cuda.synchronize()
    rows = q.shape[0]
    assert scale.shape == parent_scale.shape == (rows, 16)
    assert scale.stride() == parent_scale.stride() == (1, (rows + 3) // 4 * 4)
    assert torch.equal(q, parent_q)
    assert torch.equal(scale, parent_scale)


class TestRoutedSwiGluFp8Numerics(unittest.TestCase):
    def test_chunk_planner_uses_column_major_scale_probe(self):
        import lightllm.common.basemodel  # noqa: F401
        from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import _get_max_chunk_rows

        _get_max_chunk_rows.cache = {}
        workspace = torch.empty((1 << 20,), dtype=torch.uint8)
        rows = _get_max_chunk_rows(workspace, 8, 16, 8, 16, 1, torch.bfloat16, torch.float8_e4m3fn, 16)
        self.assertGreater(rows, 0)
        self.assertEqual(rows % 16, 0)

    @unittest.skipUnless(_cuda_hopper(), "requires SM90+ CUDA FP8")
    def test_routed_fp8_matches_parent_non_ue8m0_swiglu_quant(self):
        import lightllm.common.basemodel  # noqa: F401
        from lightllm.common.basemodel.triton_kernel.fused_moe.routed_silu_mul_quant_fp8 import (
            routed_silu_mul_quant_fp8,
        )

        for rows in (1, 3, 5):
            for kind in ("boundary", "zero_tiny", "random"):
                with self.subTest(rows=rows, kind=kind):
                    x = _input(rows, kind)
                    q = torch.empty((rows, 2048), dtype=torch.float8_e4m3fn, device="cuda")
                    scale = torch.empty((16, (rows + 3) // 4 * 4), dtype=torch.float32, device="cuda").t()[:rows]
                    routed_silu_mul_quant_fp8(x, q, scale, limit=10.0)
                    parent_q, parent_scale = _parent_quant(_parent_activation(x), ue8m0=False)
                    _assert_match(q, scale, parent_q, parent_scale)

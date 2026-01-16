"""Tests for fused QKV gating kernel."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFusedGDNGating:
    """Test fused GDN gating kernel correctness."""

    def test_fused_gating_matches_reference(self):
        """Verify fused kernel matches reference implementation."""
        from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
        from lightllm.models.qwen3next.triton_kernel.fused_qkv_gating import fused_gdn_gating_v2

        batch_size = 32
        num_heads = 8

        a = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
        A_log = torch.randn(num_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float32)

        # Reference
        g_ref, beta_ref = fused_gdn_gating(A_log, a, b, dt_bias)

        # Fused v2 with pre-allocated outputs
        g_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
        beta_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
        fused_gdn_gating_v2(a, b, A_log, dt_bias, g_out, beta_out)

        # Note: Reference kernel has precision loss (converts to bfloat16 before storing)
        # so we use slightly relaxed tolerances
        # Squeeze v2 outputs from [1, batch, heads] to [batch, heads] to match reference shape
        torch.testing.assert_close(g_out.squeeze(0), g_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(beta_out.squeeze(0), beta_ref, rtol=5e-3, atol=5e-3)

    def test_fused_gating_various_batch_sizes(self):
        """Test fused kernel with various batch sizes."""
        from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
        from lightllm.models.qwen3next.triton_kernel.fused_qkv_gating import fused_gdn_gating_v2

        num_heads = 16

        for batch_size in [1, 4, 16, 32, 64, 128]:
            a = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
            A_log = torch.randn(num_heads, device="cuda", dtype=torch.float32)
            dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float32)

            # Reference
            g_ref, beta_ref = fused_gdn_gating(A_log, a, b, dt_bias)

            # Fused v2 with pre-allocated outputs
            g_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
            beta_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
            fused_gdn_gating_v2(a, b, A_log, dt_bias, g_out, beta_out)

            # Squeeze v2 outputs from [1, batch, heads] to [batch, heads] to match reference shape
            torch.testing.assert_close(g_out.squeeze(0), g_ref, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(beta_out.squeeze(0), beta_ref, rtol=5e-3, atol=5e-3)

    def test_fused_gating_various_head_counts(self):
        """Test fused kernel with various head counts."""
        from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
        from lightllm.models.qwen3next.triton_kernel.fused_qkv_gating import fused_gdn_gating_v2

        batch_size = 32

        for num_heads in [4, 8, 16, 32, 64]:
            a = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
            A_log = torch.randn(num_heads, device="cuda", dtype=torch.float32)
            dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float32)

            # Reference
            g_ref, beta_ref = fused_gdn_gating(A_log, a, b, dt_bias)

            # Fused v2 with pre-allocated outputs
            g_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
            beta_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
            fused_gdn_gating_v2(a, b, A_log, dt_bias, g_out, beta_out)

            # Squeeze v2 outputs from [1, batch, heads] to [batch, heads] to match reference shape
            torch.testing.assert_close(g_out.squeeze(0), g_ref, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(beta_out.squeeze(0), beta_ref, rtol=5e-3, atol=5e-3)

    def test_fused_gating_float16_input(self):
        """Test fused kernel with float16 input tensors."""
        from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
        from lightllm.models.qwen3next.triton_kernel.fused_qkv_gating import fused_gdn_gating_v2

        batch_size = 32
        num_heads = 8

        a = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.float16)
        b = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.float16)
        A_log = torch.randn(num_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float32)

        # Reference
        g_ref, beta_ref = fused_gdn_gating(A_log, a, b, dt_bias)

        # Fused v2 with pre-allocated outputs
        g_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
        beta_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
        fused_gdn_gating_v2(a, b, A_log, dt_bias, g_out, beta_out)

        # Squeeze v2 outputs from [1, batch, heads] to [batch, heads] to match reference shape
        torch.testing.assert_close(g_out.squeeze(0), g_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(beta_out.squeeze(0), beta_ref, rtol=5e-3, atol=5e-3)

    def test_fused_gating_returns_same_tensors(self):
        """Test that fused_gdn_gating_v2 returns the same tensors passed in."""
        from lightllm.models.qwen3next.triton_kernel.fused_qkv_gating import fused_gdn_gating_v2

        batch_size = 32
        num_heads = 8

        a = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
        A_log = torch.randn(num_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float32)

        # Pre-allocated outputs
        g_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
        beta_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)

        # Capture data_ptr before call
        g_ptr = g_out.data_ptr()
        beta_ptr = beta_out.data_ptr()

        g_ret, beta_ret = fused_gdn_gating_v2(a, b, A_log, dt_bias, g_out, beta_out)

        # Verify same tensors returned (not new allocations)
        assert g_ret.data_ptr() == g_ptr, "g should be same tensor"
        assert beta_ret.data_ptr() == beta_ptr, "beta should be same tensor"
        assert g_ret is g_out, "Should return same tensor object"
        assert beta_ret is beta_out, "Should return same tensor object"

    def test_fused_gating_custom_beta_threshold(self):
        """Test fused kernel with custom beta and threshold parameters."""
        from lightllm.models.qwen3next.triton_kernel.fused_gdn_gating import fused_gdn_gating
        from lightllm.models.qwen3next.triton_kernel.fused_qkv_gating import fused_gdn_gating_v2

        batch_size = 32
        num_heads = 8
        beta_const = 2.0
        threshold = 10.0

        a = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(batch_size, num_heads, device="cuda", dtype=torch.bfloat16)
        A_log = torch.randn(num_heads, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float32)

        # Reference
        g_ref, beta_ref = fused_gdn_gating(A_log, a, b, dt_bias, beta=beta_const, threshold=threshold)

        # Fused v2 with pre-allocated outputs
        g_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
        beta_out = torch.empty(1, batch_size, num_heads, device="cuda", dtype=torch.float32)
        fused_gdn_gating_v2(a, b, A_log, dt_bias, g_out, beta_out, beta_const=beta_const, threshold=threshold)

        # Squeeze v2 outputs from [1, batch, heads] to [batch, heads] to match reference shape
        torch.testing.assert_close(g_out.squeeze(0), g_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(beta_out.squeeze(0), beta_ref, rtol=5e-3, atol=5e-3)

import unittest
import torch
from lightllm.models.llama.triton_kernel.fused_add_rmsnorm_inplace import fused_add_rmsnorm_inplace
from lightllm.utils.custom_kernel_utis import benchmark, error


class TestFusedAddRmsNormInplace(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.tokens = [1, 2, 3, 1024, 2048, 4096, 8192, 16384]
        self.dims = [1, 2, 3, 512, 1024, 1025, 3200, 16384, 32768]  # [512, 1024, 1032, 1536, 3200, 6144, 12800]
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def torch_add_rmsnorm(self, X, R, W):
        X.add_(R)
        return torch.nn.functional.rms_norm(X, (X.shape[1],), W, eps=1e-6)

    def test_accuracy(self):
        """Test the accuracy of fused_add_rmsnorm_inplace against torch.rmsnorm."""
        for token_num in self.tokens:
            for dim in self.dims:
                with self.subTest(shape=[token_num, dim]):
                    X = torch.randn(size=[token_num, dim], device=self.device, dtype=self.dtype)
                    _X = X.clone()
                    R = torch.randn(size=[token_num, dim], device=self.device, dtype=self.dtype)
                    _R = R.clone()
                    W = torch.randn(size=[dim], device=self.device, dtype=self.dtype)

                    r_real = self.torch_add_rmsnorm(_X, _R, W)
                    fused_add_rmsnorm_inplace(X, R, W, eps=1e-6)
                    r_pred = R
                    self.assertTrue(
                        error(r_pred, r_real) < 0.01,
                        f"Accuracy test failed for size {token_num}, {dim}. r_real={r_real}, r_pred={r_pred}",
                    )
                    print(f"{error(r_pred, r_real) = }")

                    x_real = _X
                    x_pred = X
                    self.assertTrue(
                        error(x_pred, x_real) < 0.01,
                        f"Accuracy test failed for size {token_num}, {dim}. x_real={x_real}, x_pred={x_pred}",
                    )
                    print(f"{error(x_pred, x_real) = }")

    def test_performance(self):
        """Test the performance of rmsnorm using benchmark."""
        for token_num in self.tokens:
            for dim in self.dims:
                with self.subTest(shape=[token_num, dim]):
                    X = torch.randn(size=[token_num, dim], device=self.device, dtype=self.dtype)
                    R = torch.randn(size=[token_num, dim], device=self.device, dtype=self.dtype)
                    W = torch.randn(size=[dim], device=self.device, dtype=self.dtype)

                    shape = [token_num, dim]
                    tflops = 0.0
                    benchmark(self.torch_add_rmsnorm, shape, tflops, 100, X, R, W)
                    benchmark(fused_add_rmsnorm_inplace, shape, tflops, 100, X, R, W, eps=1e-6)


if __name__ == "__main__":
    unittest.main()

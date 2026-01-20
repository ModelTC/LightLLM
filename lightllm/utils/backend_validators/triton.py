"""Triton backend validator."""

import torch
from typing import Tuple, Optional
from .base import BackendValidator


class TritonValidator(BackendValidator):
    """Validator for Triton backend.

    Triton is the fallback backend and should always be available.
    We validate that Triton JIT compilation works and produces correct results.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check if Triton is importable."""
        try:
            import triton  # noqa: F401
            import triton.language  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def validate(cls) -> Tuple[bool, Optional[str]]:
        """Validate Triton by compiling and running a simple kernel.

        We use a softmax kernel as it's a core attention operation and
        allows us to verify against PyTorch's reference implementation.
        """
        try:
            import triton
            import triton.language as tl

            @triton.jit
            def _softmax_kernel(
                input_ptr,
                output_ptr,
                n_cols,
                BLOCK_SIZE: tl.constexpr,
            ):
                row_idx = tl.program_id(0)
                col_offsets = tl.arange(0, BLOCK_SIZE)
                mask = col_offsets < n_cols

                # Load row
                row_start = row_idx * n_cols
                row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))

                # Compute softmax
                row_max = tl.max(row, axis=0)
                row = row - row_max
                numerator = tl.exp(row)
                denominator = tl.sum(numerator, axis=0)
                softmax_out = numerator / denominator

                # Store result
                tl.store(output_ptr + row_start + col_offsets, softmax_out, mask=mask)

            # Test parameters
            n_rows, n_cols = 128, 64
            BLOCK_SIZE = triton.next_power_of_2(n_cols)

            # Create test input
            x = torch.randn(n_rows, n_cols, dtype=torch.float32, device="cuda")

            # Compute ground truth
            ground_truth = torch.softmax(x, dim=-1)

            # Run Triton kernel
            output = torch.empty_like(x)
            _softmax_kernel[(n_rows,)](x, output, n_cols, BLOCK_SIZE=BLOCK_SIZE)

            torch.cuda.synchronize()

            # Verify against ground truth
            if not torch.allclose(output, ground_truth, rtol=1e-3, atol=1e-3):
                max_diff = (output - ground_truth).abs().max().item()
                return False, f"Softmax output mismatch: max diff {max_diff:.6f}"

            return True, None

        except Exception as e:
            return False, str(e)

"""Base class for attention backend validators."""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BackendValidator(ABC):
    """Abstract base class for backend validation.

    Each backend validator must:
    1. Run the backend with test inputs
    2. Compare output against ground truth (PyTorch SDPA)
    3. Return (success, error_message)
    """

    # Validation parameters
    BATCH_SIZE = 1
    SEQ_LEN = 8
    NUM_HEADS = 4
    HEAD_DIM = 64
    RTOL = 1e-2
    ATOL = 1e-2

    @classmethod
    def get_test_inputs(cls, dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate test Q, K, V tensors."""
        q = torch.randn(cls.BATCH_SIZE, cls.NUM_HEADS, cls.SEQ_LEN, cls.HEAD_DIM, dtype=dtype, device="cuda")
        k = torch.randn(cls.BATCH_SIZE, cls.NUM_HEADS, cls.SEQ_LEN, cls.HEAD_DIM, dtype=dtype, device="cuda")
        v = torch.randn(cls.BATCH_SIZE, cls.NUM_HEADS, cls.SEQ_LEN, cls.HEAD_DIM, dtype=dtype, device="cuda")
        return q, k, v

    @classmethod
    def compute_ground_truth(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute ground truth using PyTorch SDPA."""
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    @classmethod
    def check_output(cls, output: torch.Tensor, ground_truth: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """Compare output against ground truth."""
        if output.shape != ground_truth.shape:
            return False, f"Shape mismatch: {output.shape} vs {ground_truth.shape}"

        if not torch.allclose(output, ground_truth, rtol=cls.RTOL, atol=cls.ATOL):
            max_diff = (output - ground_truth).abs().max().item()
            return False, f"Output mismatch: max diff {max_diff:.6f}"

        return True, None

    @classmethod
    @abstractmethod
    def validate(cls) -> Tuple[bool, Optional[str]]:
        """Run validation and return (success, error_message)."""
        pass

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if this backend's dependencies are available."""
        pass

"""FlashInfer backend validator."""

import torch
from typing import Tuple, Optional
from .base import BackendValidator


class FlashInferValidator(BackendValidator):
    """Validator for FlashInfer backend."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if FlashInfer is importable."""
        try:
            import flashinfer  # noqa: F401

            return True
        except ImportError:
            return False

    @classmethod
    def validate(cls) -> Tuple[bool, Optional[str]]:
        """Validate FlashInfer backend with ground truth comparison."""
        try:
            import flashinfer

            dtype = torch.float16

            # Generate test inputs
            q, k, v = cls.get_test_inputs(dtype=dtype)

            # Compute ground truth
            ground_truth = cls.compute_ground_truth(q, k, v)

            batch_size, num_heads, seq_len, head_dim = q.shape

            # Reshape for FlashInfer: (batch * seq, heads, dim)
            q_flat = q.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_dim)
            k_flat = k.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_dim)
            v_flat = v.transpose(1, 2).reshape(batch_size * seq_len, num_heads, head_dim)

            # Use BatchPrefillWithRaggedKVCacheWrapper for simple validation
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

            # Create cumulative sequence lengths
            qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len, dtype=torch.int32, device="cuda")
            kv_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len, dtype=torch.int32, device="cuda")

            wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                num_heads,
                num_heads,
                head_dim,
                causal=True,
            )
            output = wrapper.run(q_flat, k_flat, v_flat)

            # Reshape back to (batch, heads, seq, dim)
            output = output.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            torch.cuda.synchronize()

            # Verify against ground truth
            return cls.check_output(output, ground_truth)

        except Exception as e:
            return False, str(e)

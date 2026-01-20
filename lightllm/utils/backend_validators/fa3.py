"""FA3 (Flash Attention 3) backend validator."""

import torch
from typing import Tuple, Optional
from .base import BackendValidator


class FA3Validator(BackendValidator):
    """Validator for FA3 backend using sgl_kernel."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if FA3 dependencies are available (Hopper GPU + sgl_kernel)."""
        try:
            from lightllm.utils.device_utils import is_hopper
            from lightllm.utils.sgl_utils import flash_attn_varlen_func

            if not is_hopper():
                return False
            if flash_attn_varlen_func is None:
                return False
            return True
        except Exception:
            return False

    @classmethod
    def validate(cls) -> Tuple[bool, Optional[str]]:
        """Validate FA3 backend with ground truth comparison."""
        try:
            from lightllm.utils.sgl_utils import flash_attn_varlen_func

            if flash_attn_varlen_func is None:
                return False, "flash_attn_varlen_func is None"

            # Use bfloat16 for FA3 (Hopper optimized)
            dtype = torch.bfloat16

            # Generate test inputs in standard attention format
            q, k, v = cls.get_test_inputs(dtype=dtype)

            # Compute ground truth
            ground_truth = cls.compute_ground_truth(q, k, v)

            # Reshape for varlen format: (total_tokens, num_heads, head_dim)
            batch_size, num_heads, seq_len, head_dim = q.shape
            total_tokens = batch_size * seq_len

            q_varlen = q.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)
            k_varlen = k.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)
            v_varlen = v.transpose(1, 2).reshape(total_tokens, num_heads, head_dim)

            # Create cumulative sequence lengths
            cu_seqlens = torch.arange(0, total_tokens + 1, seq_len, dtype=torch.int32, device="cuda")

            # Call FA3
            output = flash_attn_varlen_func(
                q=q_varlen,
                k=k_varlen,
                v=v_varlen,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=seq_len,
                max_seqlen_k=seq_len,
                softmax_scale=1.0 / (head_dim ** 0.5),
                causal=True,
            )

            # Reshape output back to (batch, heads, seq, dim)
            output = output.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

            torch.cuda.synchronize()

            # Verify against ground truth
            return cls.check_output(output, ground_truth)

        except Exception as e:
            return False, str(e)

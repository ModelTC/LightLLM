import torch

from lightllm.common.attention._environment import check_attention_environment
from lightllm.common.attention.api import UnsupportedAttention, VarlenAttentionOp
from lightllm.platform import Platform


class BaseVarlenAttentionOp(VarlenAttentionOp):
    provider = ""

    supported_platforms = frozenset({Platform.CUDA})
    dependency = ""
    min_device_capability = None
    activation_dtypes = frozenset({torch.float16, torch.bfloat16, torch.float32})
    max_head_size = None
    head_alignment = 1

    @classmethod
    def check_environment(cls, caps) -> None:
        check_attention_environment(
            caps,
            provider=cls.provider,
            supported_platforms=cls.supported_platforms,
            min_device_capability=cls.min_device_capability,
            dependency=cls.dependency,
        )

    @classmethod
    def check_support(cls, inputs, caps=None) -> None:
        q = inputs.query
        if q.ndim != 3:
            raise ValueError("varlen query must have shape [tokens, heads, dim]")
        if min(q.shape[1:]) <= 0:
            raise UnsupportedAttention("head count and dimension must be positive")
        if q.dtype not in cls.activation_dtypes:
            raise UnsupportedAttention(f"unsupported activation dtype {q.dtype}")
        if caps is not None and caps.platform not in cls.supported_platforms:
            raise UnsupportedAttention(f"{cls.provider} does not support platform {caps.platform.value}")
        if cls.max_head_size is not None and q.shape[-1] > cls.max_head_size:
            raise UnsupportedAttention(f"head dimension exceeds {cls.max_head_size}")
        if q.shape[-1] % cls.head_alignment:
            raise UnsupportedAttention(f"head dimension must be divisible by {cls.head_alignment}")

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
        q, k, v = inputs.query, inputs.key, inputs.value
        if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
            raise UnsupportedAttention("varlen self-attention requires matching [tokens, heads, dim] Q/K/V")
        if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
            raise UnsupportedAttention("Q/K/V must have the same dtype and device")
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
        cu = inputs.cu_seqlens
        if cu.ndim != 1 or cu.numel() < 2 or cu.dtype != torch.int32 or cu.device != q.device:
            raise UnsupportedAttention("cu_seqlens must be a same-device int32 vector with at least two entries")
        if inputs.max_seqlen < 0 or (q.shape[0] > 0 and inputs.max_seqlen == 0):
            raise UnsupportedAttention("nonempty attention requires a positive max_seqlen")

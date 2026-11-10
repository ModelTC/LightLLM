import importlib

import torch

from lightllm.common.attention.api import CapabilityResult, VarlenAttentionBackend
from lightllm.platform import Platform


class BaseVarlenAttentionBackend(VarlenAttentionBackend):
    provider = ""

    supported_platforms = frozenset({Platform.CUDA})
    dependency = ""
    min_device_capability = None
    activation_dtypes = frozenset({torch.float16, torch.bfloat16, torch.float32})
    max_head_size = None
    head_alignment = 1

    @classmethod
    def check_environment(cls, caps):
        if caps.platform not in cls.supported_platforms:
            return CapabilityResult.reject(f"{cls.provider} does not support platform {caps.platform.value}")
        if not caps.available:
            return CapabilityResult.reject(f"device {caps.device} is unavailable")
        if cls.min_device_capability is not None and (
            caps.capability is None or caps.capability < cls.min_device_capability
        ):
            return CapabilityResult.reject(f"requires device capability {cls.min_device_capability}")
        try:
            if cls.dependency:
                importlib.import_module(cls.dependency)
        except (ImportError, OSError) as exc:
            return CapabilityResult.reject(str(exc))
        return CapabilityResult.ok()

    @classmethod
    def check_support(cls, inputs, caps=None):
        q, k, v = inputs.query, inputs.key, inputs.value
        if q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
            return CapabilityResult.reject("varlen self-attention requires matching [tokens, heads, dim] Q/K/V")
        if q.device != k.device or q.device != v.device or q.dtype != k.dtype or q.dtype != v.dtype:
            return CapabilityResult.reject("Q/K/V must have the same dtype and device")
        if min(q.shape[1:]) <= 0:
            return CapabilityResult.reject("head count and dimension must be positive")
        if q.dtype not in cls.activation_dtypes:
            return CapabilityResult.reject(f"unsupported activation dtype {q.dtype}")
        if caps is not None and caps.platform not in cls.supported_platforms:
            return CapabilityResult.reject(f"{cls.provider} does not support platform {caps.platform.value}")
        if cls.max_head_size is not None and q.shape[-1] > cls.max_head_size:
            return CapabilityResult.reject(f"head dimension exceeds {cls.max_head_size}")
        if q.shape[-1] % cls.head_alignment:
            return CapabilityResult.reject(f"head dimension must be divisible by {cls.head_alignment}")
        cu = inputs.cu_seqlens
        if cu.ndim != 1 or cu.numel() < 2 or cu.dtype != torch.int32 or cu.device != q.device:
            return CapabilityResult.reject("cu_seqlens must be a same-device int32 vector with at least two entries")
        if inputs.max_seqlen < 0 or (q.shape[0] > 0 and inputs.max_seqlen == 0):
            return CapabilityResult.reject("nonempty attention requires a positive max_seqlen")
        return CapabilityResult.ok()

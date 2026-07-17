from lightllm.utils.device_utils import is_musa

if is_musa():
    import torchada  # noqa: F401
else:
    import torch

    torch._C._accelerator_setAllocatorSettings("expandable_segments:True")

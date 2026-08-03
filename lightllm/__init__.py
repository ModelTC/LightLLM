import os

os.environ.setdefault("EP_REUSE_NCCL_COMM", "0")

from lightllm.utils.device_utils import is_musa

if is_musa():
    import torchada  # noqa: F401

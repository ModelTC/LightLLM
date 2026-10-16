import torch
from lightllm.utils.log_utils import init_logger

from lightllm.backends import get_device_runtime

_device_runtime = get_device_runtime()


logger = init_logger(__name__)
try:
    if _device_runtime.is_available() and not _device_runtime.is_initialized():
        _device_runtime.init()
    from vllm import _custom_ops as ops

    vllm_ops = ops
    HAS_VLLM = True
    cutlass_scaled_mm = torch.ops._C.cutlass_scaled_mm

except:
    HAS_VLLM = False
    cutlass_scaled_mm = None
    vllm_ops = None
    logger.warning(
        "vllm is not installed, you can't use the api of it. \
                   You can solve it by running `pip install vllm`."
    )

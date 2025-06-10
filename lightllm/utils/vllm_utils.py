import torch
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        torch.cuda.init()
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

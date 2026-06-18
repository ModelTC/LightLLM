from lightllm.platform.backends.cuda import CudaBackend
from lightllm.platform.base.registry import register_platform


@register_platform("maca", op_fallback=("cuda_like",), sampling_fallback=("cuda_like",))
class MacaBackend(CudaBackend):
    pass

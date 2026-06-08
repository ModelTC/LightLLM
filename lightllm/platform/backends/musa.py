from lightllm.platform.backends.cuda import CudaBackend
from lightllm.platform.base.registry import register_platform


@register_platform("musa", op_fallback=("cuda_like",))
class MusaBackend(CudaBackend):
    pass

from lightllm.platform.base.backend import HardwareBackend
from lightllm.platform.base.registry import register_platform
from lightllm.platform.backends.cuda_like.runtime import CudaLikeRuntime
from lightllm.platform.backends.cuda_like.graph import CudaLikeGraph


class CudaLikeBackend(HardwareBackend):
    def __init__(self) -> None:
        super().__init__(CudaLikeRuntime(), CudaLikeGraph())


@register_platform("cuda")
class CudaBackend(CudaLikeBackend):
    pass

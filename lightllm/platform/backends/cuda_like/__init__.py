from lightllm.platform.base.backend import HardwareBackend
from lightllm.platform.backends.cuda_like.ops import CUDA_LIKE_OPS
from lightllm.platform.backends.cuda_like.runtime import CudaLikeRuntime
from lightllm.platform.backends.cuda_like.graph import CudaLikeGraph
from lightllm.platform.plugin.ops import build_platform_ops


class CudaLikeBackend(HardwareBackend):
    def __init__(self) -> None:
        super().__init__(
            runtime=CudaLikeRuntime(),
            graph=CudaLikeGraph(),
            ops=build_platform_ops(self.platform_name, CUDA_LIKE_OPS),
        )


class CudaBackend(CudaLikeBackend):
    platform_name = "cuda"


class MusaBackend(CudaLikeBackend):
    platform_name = "musa"

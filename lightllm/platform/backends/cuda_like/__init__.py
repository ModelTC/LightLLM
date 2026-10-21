from lightllm.platform.base.backend import HardwareBackend
from lightllm.platform.backends.cuda_like.runtime import CudaLikeRuntime
from lightllm.platform.backends.cuda_like.graph import CudaLikeGraph
from lightllm.platform.plugin.ops import build_platform_ops


def build_cuda_like_ops(platform_name: str):
    from lightllm.platform.backends.cuda_like.ops import CUDA_LIKE_OPS

    return build_platform_ops(platform_name, CUDA_LIKE_OPS)


class CudaLikeBackend(HardwareBackend):
    def __init__(self) -> None:
        super().__init__(
            runtime=CudaLikeRuntime(),
            graph=CudaLikeGraph(),
            ops=lambda: build_cuda_like_ops(self.platform_name),
        )


class CudaBackend(CudaLikeBackend):
    platform_name = "cuda"


class MusaBackend(CudaLikeBackend):
    platform_name = "musa"

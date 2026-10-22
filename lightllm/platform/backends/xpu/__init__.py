from lightllm.platform.base.backend import HardwareBackend
from lightllm.platform.backends.cuda_like import build_cuda_like_ops
from lightllm.platform.backends.xpu.graph import XpuGraph
from lightllm.platform.backends.xpu.runtime import XpuRuntime


class XpuBackend(HardwareBackend):
    platform_name = "xpu"

    def __init__(self) -> None:
        # The Triton kernels behind the cuda_like ops run unchanged on Intel Triton.
        super().__init__(
            runtime=XpuRuntime(),
            graph=XpuGraph(),
            ops=lambda: build_cuda_like_ops(self.platform_name),
        )

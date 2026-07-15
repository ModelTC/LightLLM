from lightllm.platform.base.registry import Backend, register_platform
from lightllm.platform.graph.ascend import AscendGraphBackend
from lightllm.platform.graph.cuda import CudaGraphBackend
from lightllm.platform.runtime.ascend import AscendRuntime
from lightllm.platform.runtime.cuda import CudaRuntime


@register_platform("cuda", ops_fallback=("cuda_like",))
class CudaBackend(Backend):

    def __init__(self) -> None:
        super().__init__(CudaRuntime(), CudaGraphBackend())


@register_platform("ascend", ops_fallback=("ascend",), sampling_fallback=("cuda_like",))
class AscendBackend(Backend):

    def __init__(self) -> None:
        super().__init__(AscendRuntime(), AscendGraphBackend())


@register_platform("maca", ops_fallback=("cuda_like",), sampling_fallback=("cuda_like",))
class MacaBackend(CudaBackend):
    pass


@register_platform("musa", ops_fallback=("cuda_like",), sampling_fallback=("cuda_like",))
class MusaBackend(CudaBackend):
    pass

from lightllm.platform.base.ops import build_ops
from lightllm.platform.base.registry import Backend, register_platform
from lightllm.platform.base.sampling import build_sampling
from lightllm.platform.graph.cuda import CudaGraphBackend
from lightllm.platform.runtime.cuda import CudaRuntime


@register_platform("cuda", op_fallback=("cuda_like",))
class CudaBackend(Backend):

    def __init__(self) -> None:
        self._runtime = CudaRuntime()
        self._graph = CudaGraphBackend()
        self._ops = build_ops(self.platform_name)
        self._sampling = build_sampling(self.platform_name)

from lightllm.platform.base.graph import BackendGraph
from lightllm.platform.base.ops import BackendOps
from lightllm.platform.base.registry import Backend, register_backend
from lightllm.platform.base.runtime import BackendRuntime
from lightllm.platform.graph.cuda import CudaGraphBackend
from lightllm.platform.ops.cuda import CudaOps
from lightllm.platform.runtime.cuda import CudaRuntime


@register_backend("cuda")
class CudaBackend(Backend):

    def __init__(self) -> None:
        self._runtime = CudaRuntime()
        self._graph = CudaGraphBackend()
        self._ops = CudaOps()

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def runtime(self) -> BackendRuntime:
        return self._runtime

    @property
    def graph(self) -> BackendGraph:
        return self._graph

    @property
    def ops(self) -> BackendOps:
        return self._ops

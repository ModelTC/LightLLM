from lightllm.platform.base.graph import BackendGraph
from lightllm.platform.base.ops import BackendOps
from lightllm.platform.base.registry import Backend, register_backend
from lightllm.platform.base.runtime import BackendRuntime
from lightllm.platform.graph.ascend import AscendGraphBackend
from lightllm.platform.ops.ascend import AscendOps
from lightllm.platform.runtime.ascend import AscendRuntime


@register_backend("ascend")
class AscendBackend(Backend):

    def __init__(self) -> None:
        self._runtime = AscendRuntime()
        self._graph = AscendGraphBackend()
        self._ops = AscendOps()

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def runtime(self) -> BackendRuntime:
        return self._runtime

    @property
    def graph(self) -> BackendGraph:
        return self._graph

    @property
    def ops(self) -> BackendOps:
        return self._ops

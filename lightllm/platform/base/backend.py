from lightllm.platform.base.graph import HardwareBackendGraph
from lightllm.platform.base.runtime import HardwareBackendRuntime
from lightllm.platform.ops import PlatformOps


class HardwareBackend:
    platform_name: str

    def __init__(
        self,
        runtime: HardwareBackendRuntime,
        graph: HardwareBackendGraph,
        ops: PlatformOps,
    ) -> None:
        self._runtime = runtime
        self._graph = graph
        self._ops = ops

    @property
    def name(self) -> str:
        return self.platform_name

    @property
    def runtime(self) -> HardwareBackendRuntime:
        return self._runtime

    @property
    def graph(self) -> HardwareBackendGraph:
        return self._graph

    @property
    def ops(self) -> PlatformOps:
        return self._ops

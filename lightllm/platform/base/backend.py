from abc import ABC

from lightllm.platform.base.graph import HardwareBackendGraph
from lightllm.platform.base.runtime import HardwareBackendRuntime


class HardwareBackend(ABC):
    platform_name: str

    def __init__(self, runtime: HardwareBackendRuntime, graph: HardwareBackendGraph) -> None:
        self._runtime = runtime
        self._graph = graph

    @property
    def name(self) -> str:
        return self.platform_name

    @property
    def runtime(self) -> HardwareBackendRuntime:
        return self._runtime

    @property
    def graph(self) -> HardwareBackendGraph:
        return self._graph

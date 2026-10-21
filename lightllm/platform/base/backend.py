from typing import Callable, Union

from lightllm.platform.base.graph import HardwareBackendGraph
from lightllm.platform.base.runtime import HardwareBackendRuntime
from lightllm.platform.ops import PlatformOps


class HardwareBackend:
    platform_name: str

    def __init__(
        self,
        runtime: HardwareBackendRuntime,
        graph: HardwareBackendGraph,
        ops: Union[PlatformOps, Callable[[], PlatformOps]],
    ) -> None:
        self._runtime = runtime
        self._graph = graph
        # ops may be a factory, the op tables import the kernels and are only built on first use, so
        # processes that only need the runtime (device type, dist backend) stay light.
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
        if not isinstance(self._ops, PlatformOps):
            self._ops = self._ops()
        return self._ops

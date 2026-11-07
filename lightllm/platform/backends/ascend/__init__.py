import lightllm.platform.backends.ascend.ops  # noqa: F401
from lightllm.platform.base.backend import HardwareBackend
from lightllm.platform.backends.ascend.graph import AscendGraph
from lightllm.platform.backends.ascend.runtime import AscendRuntime
from lightllm.platform.plugin.ops import build_platform_ops


class AscendBackend(HardwareBackend):
    platform_name = "ascend"

    def __init__(self) -> None:
        super().__init__(
            runtime=AscendRuntime(),
            graph=AscendGraph(),
            ops=build_platform_ops(self.platform_name),
        )

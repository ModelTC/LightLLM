from lightllm.platform.base.ops import build_ops
from lightllm.platform.base.registry import Backend, register_platform
from lightllm.platform.base.sampling import build_sampling
from lightllm.platform.graph.ascend import AscendGraphBackend
from lightllm.platform.runtime.ascend import AscendRuntime


@register_platform("ascend", op_fallback=("ascend",), sampling_fallback=("cuda_like",))
class AscendBackend(Backend):

    def __init__(self) -> None:
        self._runtime = AscendRuntime()
        self._graph = AscendGraphBackend()
        self._ops = build_ops(self.platform_name)
        self._sampling = build_sampling(self.platform_name)

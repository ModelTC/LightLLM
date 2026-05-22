from lightllm.platform.base.registry import Backend, register_backend
from lightllm.platform.base.runtime import BackendRuntime
from lightllm.platform.runtime.cuda import CudaRuntime


@register_backend("cuda")
class CudaBackend(Backend):

    def __init__(self) -> None:
        self._runtime = CudaRuntime()

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def runtime(self) -> BackendRuntime:
        return self._runtime

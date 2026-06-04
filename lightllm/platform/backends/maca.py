from lightllm.platform.backends import CudaBackend  # noqa: F401
from lightllm.platform.base.registry import Backend, register_backend


@register_backend("maca")
class MacaBackend(CudaBackend):

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        return "maca"


import torch
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from lightllm.utils.device_utils import get_platform
from lightllm.backends import get_operator_backend
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PlatformAwareOp(ABC):
    """
    platform aware op base class,
    automatically route to the corresponding implementation method according to the platform.
    """

    def __init__(self):
        args = get_env_start_args()
        self.platform = get_platform(args.hardware_platform)
        self.enable_torch_fallback = args.enable_torch_fallback
        self.enable_triton_fallback = args.enable_triton_fallback
        self._forward = self._route_forward()

    def _route_forward(self) -> Callable:

        return get_operator_backend().select_weight_forward(self)

    @abstractmethod
    def _native_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("default forward must implement this method")

    @abstractmethod
    def _cuda_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("cuda forward must implement this method")

    # Since Triton may be compatible with all hardware platforms in the future,
    # so provide triton implementation as a fallback for all hardware platforms
    def _triton_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("triton forward must implement this method")

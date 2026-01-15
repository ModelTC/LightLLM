import torch
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from lightllm.utils.device_utils import get_platform, Platform
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

        method_name_map = {
            Platform.CUDA: "_cuda_forward",
            Platform.ASCEND: "_ascend_forward",
            Platform.CAMBRICON: "_cambricon_forward",
            Platform.MUSA: "_musa_forward",
            Platform.ROCM: "_rocm_forward",
            Platform.CPU: "_cpu_forward",
        }

        method_name = method_name_map.get(self.platform)
        if method_name and hasattr(self, method_name):
            method = getattr(self, method_name)
            if callable(method):
                return method

        if self.enable_triton_fallback:
            if hasattr(self, "_triton_forward"):
                return self._triton_forward
            logger.warning(
                f"No triton implementation found for {self.__class__.__name__} on {self.platform.name} platform. "
                f"Please implement {self.__class__.__name__}_{self.platform.name}_triton_forward method, "
                f"or set --enable_torch_fallback to use default implementation."
            )

        if self.enable_torch_fallback:
            return self._native_forward

        # if no implementation found, raise error
        raise NotImplementedError(
            f"No implementation found for {self.__class__.__name__} on {self.platform.name} platform. "
            f"Please implement {self.__class__.__name__}_{self.platform.name}_forward method, "
            f"or set --enable_torch_fallback to use default implementation."
        )

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

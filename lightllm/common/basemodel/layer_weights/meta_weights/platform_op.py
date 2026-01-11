import torch
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
from lightllm.utils.device_utils import get_platform, Platform
from lightllm.utils.envs_utils import get_env_start_args


class PlatformAwareOp(ABC):
    """
    platform aware op base class,
    automatically route to the corresponding implementation method according to the platform.
    """

    def __init__(self):
        args = get_env_start_args()
        self.platform = get_platform(args.hardware_platform)
        self.enable_torch_naive = args.enable_torch_naive
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

        if self.enable_torch_naive:
            return self._native_forward

        # 如果都没有，抛出异常
        raise NotImplementedError(
            f"No implementation found for platform {self.platform.name}. "
            f"Please implement _{self.platform.name}_forward method, "
            f"or set --enable_torch_naive to use default implementation."
        )

    @abstractmethod
    def _native_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("default forward must implement this method")

    @abstractmethod
    def _cuda_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("cuda forward must implement this method")

from dataclasses import dataclass
from lightllm.common.execution.contracts.device import DeviceRuntime
from lightllm.common.execution.contracts.state import FormatSpec
from lightllm.common.execution.registry import ProviderRegistry
from importlib import import_module
import os


@dataclass
class ExecutionContext:
    device: DeviceRuntime
    providers: ProviderRegistry
    formats: dict[str, FormatSpec] | None = None
    communication: object = None


def create_context(index, *, device=None, providers=None, formats=None, config=None):
    factory = os.environ.get("LIGHTLLM_EXECUTION_FACTORY")
    if factory and device is None:
        module, separator, name = factory.partition(":")
        if not separator or not module or not name:
            raise ValueError("LIGHTLLM_EXECUTION_FACTORY must be module:factory")
        context = getattr(import_module(module), name)(index=index, config=config)
        if not isinstance(context, ExecutionContext):
            raise TypeError("execution factory must return ExecutionContext")
        if providers is not None:
            context.providers = providers
        if formats is not None:
            context.formats = formats
        return context
    if device is None:
        platform = getattr(config, "hardware_platform", "cuda")
        if platform != "cuda":
            raise ValueError(f"platform {platform} requires an execution factory or an explicit runtime")
        from .cuda import CudaRuntime

        device = CudaRuntime(index)
    if providers is None:
        if not device.caps.is_nvidia:
            raise ValueError("an explicit provider registry is required for this platform")
        from lightllm.common.execution.registry import default_providers

        providers = default_providers()
    return ExecutionContext(device, providers, formats)

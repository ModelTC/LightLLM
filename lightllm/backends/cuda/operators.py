"""CUDA operator provider, using the pre-existing LightLLM implementations."""

from functools import lru_cache
from importlib import import_module

from lightllm.backends.operators import OperatorBackend
from lightllm.utils.device_utils import Platform
from lightllm.utils.log_utils import init_logger
from .bindings import KERNEL_BINDINGS

logger = init_logger("lightllm.common.basemodel.layer_weights.meta_weights.platform_op")


class CudaOperatorBackend(OperatorBackend):
    name = "cuda"
    supports_autotune = True

    def resolve(self, name):
        if name == "sampling.gather":
            import torch

            return torch.gather
        if name == "tensor_copy_3dim":
            from .tensor_copy import tensor_copy_3dim

            return tensor_copy_3dim
        try:
            module_name, attribute = KERNEL_BINDINGS[name]
        except KeyError:
            raise NotImplementedError(f"CUDA operator {name!r} is not registered") from None
        module = import_module(module_name)
        try:
            return getattr(module, attribute)
        except AttributeError:
            # Preserve the ImportError contract of the original `from ... import`.
            raise ImportError(f"cannot import name {attribute!r} from {module_name!r}") from None

    @lru_cache(maxsize=None)
    def selection_module(self, family):
        if family not in ("attention", "linear_attention", "vision_attention"):
            raise ValueError(f"Unknown attention family: {family!r}")
        return import_module(f"{__package__}.{family}")

    def select_weight_forward(self, op):
        method_name_map = {
            Platform.CUDA: "_cuda_forward",
            Platform.ASCEND: "_ascend_forward",
            Platform.CAMBRICON: "_cambricon_forward",
            Platform.MUSA: "_musa_forward",
            Platform.ROCM: "_rocm_forward",
            Platform.CPU: "_cpu_forward",
        }

        method_name = method_name_map.get(op.platform)
        if method_name and hasattr(op, method_name):
            method = getattr(op, method_name)
            if callable(method):
                return method

        if op.enable_triton_fallback:
            if hasattr(op, "_triton_forward"):
                return op._triton_forward
            logger.warning(
                f"No triton implementation found for {op.__class__.__name__} on {op.platform.name} platform. "
                f"Please implement {op.__class__.__name__}_{op.platform.name}_triton_forward method, "
                f"or set --enable_torch_fallback to use default implementation."
            )

        if op.enable_torch_fallback:
            return op._native_forward

        # if no implementation found, raise error
        raise NotImplementedError(
            f"No implementation found for {op.__class__.__name__} on {op.platform.name} platform. "
            f"Please implement {op.__class__.__name__}_{op.platform.name}_forward method, "
            f"or set --enable_torch_fallback to use default implementation."
        )

    def initialize_linear_attention(self):
        import torch
        import triton

        def _triton_allocator(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(_triton_allocator)

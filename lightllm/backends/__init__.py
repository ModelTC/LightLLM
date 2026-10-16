"""Independent device-runtime and operator-backend boundaries.

CUDA remains the default. Importing these APIs does not import CUDA kernels or
probe accelerator availability. Select providers before importing model code.
"""

import os
from typing import TYPE_CHECKING

from ._registry import BackendRegistry

if TYPE_CHECKING:
    from .runtime import DeviceRuntime
    from .operators import OperatorBackend


def _cuda_runtime():
    from .cuda.runtime import CudaRuntime

    return CudaRuntime()


def _cuda_operators():
    from .cuda.operators import CudaOperatorBackend

    return CudaOperatorBackend()


_runtimes = BackendRegistry("device runtime", _cuda_runtime)
_operators = BackendRegistry("operator", _cuda_operators)


def register_device_runtime(name, factory):
    _runtimes.register(name, factory)


def select_device_runtime(name):
    _runtimes.select(name)


def get_device_runtime() -> "DeviceRuntime":
    return _runtimes.get()


def register_operator_backend(name, factory):
    _operators.register(name, factory)


def select_operator_backend(name):
    _operators.select(name)


def get_operator_backend() -> "OperatorBackend":
    return _operators.get()


def _metal_runtime():
    from .metal.runtime import MetalRuntime

    return MetalRuntime()


def _metal_operators():
    from .metal.operators import MetalOperatorBackend

    return MetalOperatorBackend()


# The environment is inherited by multiprocessing's spawn workers. Selection
# happens before model modules bind their runtime and operator implementations.
register_device_runtime("metal", _metal_runtime)
register_operator_backend("metal", _metal_operators)
_backend = os.environ.get("LIGHTLLM_BACKEND", "cuda")
select_device_runtime(_backend)
select_operator_backend(_backend)


def configure_backends(platform):
    """Select providers before importing inference modules, including in workers."""
    name = "metal" if platform == "metal" else "cuda"
    select_device_runtime(name)
    select_operator_backend(name)
    os.environ["LIGHTLLM_BACKEND"] = name

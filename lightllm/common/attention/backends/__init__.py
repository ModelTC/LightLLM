import os

import torch


def require_positive(layer_config: object, *fields: str) -> None:
    """Reject a layer config whose named integer fields are not positive."""
    invalid = {name: getattr(layer_config, name) for name in fields if getattr(layer_config, name) <= 0}
    if invalid:
        raise ValueError(f"{type(layer_config).__name__} fields must be positive, got {invalid}")


def set_flashinfer_envs() -> None:
    """Point FlashInfer's JIT compilation at the current device's architecture."""
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{arch}{'+PTX' if arch == '9.0' else ''}"

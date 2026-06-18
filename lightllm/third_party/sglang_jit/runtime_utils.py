import torch


def is_hip() -> bool:
    return torch.version.hip is not None

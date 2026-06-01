import torch
from typing import Callable, Optional, Tuple, Union


def ensure_out(
    out: Optional[torch.Tensor],
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    alloc_func: Callable = torch.empty,
    contiguous: bool = True,
) -> torch.Tensor:
    if out is None:
        return alloc_func(shape, dtype=dtype, device=device)

    expected_shape = tuple(shape)
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"out.shape {tuple(out.shape)} != expected {expected_shape}")
    if out.dtype != dtype:
        raise ValueError(f"out.dtype {out.dtype} != expected {dtype}")

    expected_device = torch.device(device)
    if out.device != expected_device:
        raise ValueError(f"out.device {out.device} != expected {expected_device}")

    if contiguous and not out.is_contiguous():
        raise ValueError("out must be contiguous")

    return out

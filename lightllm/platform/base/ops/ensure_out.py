import torch
from typing import Any, Callable, Optional, Tuple, TypedDict, Union

# (int, int, ...) or (("tensor_name", dim_index), ...)
OutShapeSpec = Union[Tuple[int, ...], Tuple[Tuple[str, int], ...]]
# torch.dtype or "tensor_name"
OutDtypeSpec = Union[torch.dtype, str]
# torch.device or "tensor_name"
OutDeviceSpec = Union[torch.device, str]


class AutoOutSpec(TypedDict, total=False):
    input_name: str
    out_shape: OutShapeSpec
    out_dtype: OutDtypeSpec
    out_device: OutDeviceSpec


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

    if tuple(out.shape) != tuple(shape):
        raise ValueError(f"out.shape {tuple(out.shape)} != expected {tuple(shape)}")
    if out.dtype != dtype:
        raise ValueError(f"out.dtype {out.dtype} != expected {dtype}")
    if out.device != torch.device(device):
        raise ValueError(f"out.device {out.device} != expected {device}")
    if contiguous and not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


def _is_literal_shape(spec: OutShapeSpec) -> bool:
    return all(isinstance(dim, int) for dim in spec)


def _resolve_shape(spec: OutShapeSpec, kwargs: dict) -> tuple[int, ...]:
    if _is_literal_shape(spec):
        return tuple(spec)
    dims: list[int] = []
    for name, dim in spec:
        if name not in kwargs:
            raise ValueError(
                f"out_shape references {name!r} but kwargs keys are {list(kwargs.keys())}"
            )
        tensor = kwargs[name]
        try:
            dims.append(tensor.shape[dim])
        except IndexError as exc:
            raise ValueError(
                f"out_shape ({name!r}, {dim}) is invalid for tensor shape {tuple(tensor.shape)}"
            ) from exc
    return tuple(dims)


def _resolve_dtype(spec: OutDtypeSpec, kwargs: dict) -> torch.dtype:
    if isinstance(spec, torch.dtype):
        return spec
    return kwargs[spec].dtype


def _resolve_device(spec: OutDeviceSpec, kwargs: dict) -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if spec in kwargs:
        return kwargs[spec].device
    return torch.device(spec)


def _is_out_fully_specified(config: AutoOutSpec) -> bool:
    return (
        config.get("out_shape") is not None
        and config.get("out_dtype") is not None
        and config.get("out_device") is not None
    )


def _get_base_tensor(config: AutoOutSpec, kwargs: dict) -> tuple[torch.Tensor, str]:
    input_name = config["input_name"]
    if input_name not in kwargs:
        raise ValueError(
            f"input_name '{input_name}' not found in kwargs, available keys: {list(kwargs.keys())}"
        )

    tensor = kwargs[input_name]
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"kwargs['{input_name}'] must be a torch.Tensor, got {type(tensor)}")
    return tensor, input_name


def _resolve_out_spec(config: AutoOutSpec, kwargs: dict) -> tuple[tuple[int, ...], torch.dtype, torch.device]:
    # If out_shape, out_dtype, and out_device are all specified, then input_name is optional
    if _is_out_fully_specified(config):
        shape = _resolve_shape(config["out_shape"], kwargs)
        dtype = _resolve_dtype(config["out_dtype"], kwargs)
        device = _resolve_device(config["out_device"], kwargs)
        return shape, dtype, device
    # If out_shape, out_dtype, and out_device are not all specified, then input_name is required
    if "input_name" not in config:
        raise ValueError(
            "input_name is required when out_shape, out_dtype, and out_device are not all specified"
        )
    # Get the base tensor and input_name
    base_tensor, input_name = _get_base_tensor(config, kwargs)
    out_shape = config.get("out_shape")
    # If out_shape is not specified, use the shape of the base tensor
    if out_shape is None:
        shape: tuple[int, ...] = tuple(base_tensor.shape)
        dtype = base_tensor.dtype
        device = base_tensor.device
    else:
        shape = _resolve_shape(out_shape, kwargs)
        dtype = _resolve_dtype(config.get("out_dtype", input_name), kwargs)
        device = _resolve_device(config.get("out_device", input_name), kwargs)

    return shape, dtype, device


def wrap_with_out(config: AutoOutSpec, impl: Callable) -> Callable:

    def public(*, out: Optional[torch.Tensor] = None, alloc_func: Callable = torch.empty, **kwargs: Any):
        shape, dtype, device = _resolve_out_spec(config, kwargs)
        out = ensure_out(out, shape=shape, dtype=dtype, device=device, alloc_func=alloc_func)
        return impl(out=out, **kwargs)

    public.__name__ = impl.__name__
    public.__doc__ = impl.__doc__
    return public

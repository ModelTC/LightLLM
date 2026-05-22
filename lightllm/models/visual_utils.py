import torch
import torch.nn as nn
from typing import List, Optional, Sequence, Union
from lightllm.utils.device_utils import get_target_device
from lightllm.utils.dist_utils import get_current_device_id


def default_infer_dtype(device_id: Optional[int] = None) -> torch.dtype:
    device = get_target_device(device_id)
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    return torch.bfloat16


def _tensor_to_infer_device(
    tensor: torch.Tensor,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = True,
) -> torch.Tensor:
    out = tensor.to(device=device, non_blocking=non_blocking)
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _resolve_move_dtypes(
    n: int,
    dtype: Optional[Union[torch.dtype, Sequence[Optional[torch.dtype]]]],
) -> List[Optional[torch.dtype]]:
    if dtype is None:
        return [None] * n
    if isinstance(dtype, torch.dtype):
        return [dtype] * n
    if isinstance(dtype, (list, tuple)):
        dtypes = list(dtype)
        if len(dtypes) != n:
            raise ValueError(f"dtype length {len(dtypes)} must match number of tensors ({n})")
        return dtypes
    raise TypeError(f"dtype must be torch.dtype or a sequence, got {type(dtype)!r}")


class VisualDeviceMixin:
    device_id: Optional[int] = None
    target_device: Optional[torch.device] = None

    def setup_device(self, device_id: Optional[int] = None):
        self.device_id = device_id if device_id is not None else get_current_device_id()
        self.target_device = get_target_device(self.device_id)
        if isinstance(self, nn.Module):
            self.to(device=self.target_device)
        else:
            self._setup_device_non_module()
        return self

    def _device_module_attrs(self) -> Sequence[str]:
        """ The attributes that are nn.Modules, e.g. vision_tower, audio, model """
        return ()

    def _device_tensor_dict_attrs(self) -> Sequence[str]:
        # attributes that are dict[str, Tensor], e.g. projector_weights
        """ The attributes that are dict[str, Tensor], e.g. projector_weights """
        return ()

    def _move_module_attr(self, name: str) -> None:
        mod = getattr(self, name, None)
        if isinstance(mod, nn.Module):
            setattr(self, name, mod.to(device=self.target_device))

    def _move_tensor_dict_attr(self, name: str) -> None:
        weights = getattr(self, name, None)
        if isinstance(weights, dict):
            for k, v in list(weights.items()):
                weights[k] = v.to(device=self.target_device)

    def _setup_device_non_module(self):
        for name in self._device_module_attrs():
            self._move_module_attr(name)
        for name in self._device_tensor_dict_attrs():
            self._move_tensor_dict_attr(name)

    @property
    def infer_device(self) -> torch.device:
        if self.target_device is not None:
            return self.target_device
        if isinstance(self, nn.Module):
            for param in self.parameters():
                if param.numel() > 0:
                    return param.device
        raise RuntimeError(f"{type(self).__name__}: call setup_device() before inference")

    def move_to_infer_device(
        self,
        *tensors: torch.Tensor,
        dtype: Optional[Union[torch.dtype, Sequence[Optional[torch.dtype]]]] = None,
        non_blocking: bool = True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not tensors:
            raise ValueError("move_to_infer_device() requires at least one tensor")
        device = self.infer_device
        dtypes = _resolve_move_dtypes(len(tensors), dtype)
        if len(tensors) == 1:
            return _tensor_to_infer_device(tensors[0], device, dtype=dtypes[0], non_blocking=non_blocking)
        return [
            _tensor_to_infer_device(t, device, dtype=dt, non_blocking=non_blocking)
            for t, dt in zip(tensors, dtypes)
        ]

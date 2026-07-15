import torch
from typing import Any, Callable, Optional, Tuple, Union


def ensure_out(
    out: Optional[torch.Tensor],
    *,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    alloc_func: Callable = torch.empty,
    contiguous: bool = True,
) -> torch.Tensor:
    # 如果 out 为 None 则按照输入信息分配
    if out is None:
        return alloc_func(shape, dtype=dtype, device=device)

    # 校验 out 是否符合预期
    if tuple(out.shape) != tuple(shape):
        raise ValueError(f"out.shape {tuple(out.shape)} != expected {tuple(shape)}")
    if out.dtype != dtype:
        raise ValueError(f"out.dtype {out.dtype} != expected {dtype}")
    if out.device != torch.device(device):
        raise ValueError(f"out.device {out.device} != expected {torch.device(device)}")
    if contiguous and not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


def out_like(tensor_name: str) -> Callable:
    # 构造 out_spec：输出 buffer 的 shape/dtype/device 与 kwargs 中某个 tensor 一致

    def spec(kwargs: dict[str, Any]) -> tuple:
        t = kwargs[tensor_name]
        return tuple(t.shape), t.dtype, t.device

    return spec


def wrap_out(impl: Callable, out_spec: Callable) -> Callable:
    # 包装算子：先按 out_spec 得到期望规格，ensure_out 分配或校验 out，再调用真正的实现

    def public(*, out: Optional[torch.Tensor] = None, alloc_func: Callable = torch.empty, **kwargs: Any):
        # 从本次调用参数解析出 shape / dtype / device
        shape, dtype, device = out_spec(kwargs)
        # 无 out 则分配，有 out 则校验
        out = ensure_out(out, shape=shape, dtype=dtype, device=device, alloc_func=alloc_func)
        return impl(out=out, **kwargs)

    # 保留原函数名和文档，便于调试与 help
    public.__name__ = impl.__name__
    public.__doc__ = impl.__doc__
    return public

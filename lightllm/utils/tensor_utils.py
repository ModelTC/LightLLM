import torch
from lightllm.platform import get_device_module, get_device_type


def ptr_to_tensor(device_ptr: int, nbytes: int) -> torch.Tensor:
    if get_device_type() != "cuda":
        return _ptr_to_tensor_no_cupy(device_ptr, nbytes)

    import cupy as cp

    mem = cp.cuda.UnownedMemory(device_ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    arr = cp.ndarray((nbytes,), dtype=cp.uint8, memptr=memptr)
    return torch.as_tensor(arr, dtype=torch.uint8, device="cuda")


def _ptr_to_tensor_no_cupy(device_ptr: int, nbytes: int) -> torch.Tensor:
    """Non-owning uint8 tensor over device memory for platforms without cupy (xpu)."""
    # XPU USM pointers have bit 63 set, the binding takes a signed int64.
    if device_ptr >= 1 << 63:
        device_ptr -= 1 << 64
    device = torch.device(get_device_type(), get_device_module().current_device())
    storage = torch._C._construct_storage_from_data_pointer(device_ptr, device, nbytes)
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage)


def tensor_to_no_ref_tensor(origin_tensor: torch.Tensor) -> torch.Tensor:
    """将tensor转换为无引用计数的tensor，避免cuda graph捕获时的引用计数问题,
    导致 prefill cuda graph 的中间 tensor 无法释放和共享
    Args:
        tensor (torch.Tensor): 输入tensor
    Returns:
        torch.Tensor: 无引用计数的tensor
    """
    assert origin_tensor.is_contiguous(), "Only support contiguous tensor"
    device_ptr = origin_tensor.data_ptr()
    nbytes = origin_tensor.numel() * origin_tensor.element_size()
    no_ref_tensor = ptr_to_tensor(device_ptr, nbytes).view(dtype=origin_tensor.dtype).view(size=origin_tensor.shape)
    return no_ref_tensor

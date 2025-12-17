import torch


def ptr_to_tensor(device_ptr: int, nbytes: int) -> torch.Tensor:
    import cupy as cp

    mem = cp.cuda.UnownedMemory(device_ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    arr = cp.ndarray((nbytes,), dtype=cp.uint8, memptr=memptr)
    return torch.as_tensor(arr, dtype=torch.uint8, device="cuda")


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

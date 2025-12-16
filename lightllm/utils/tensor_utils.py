import torch


def ptr_to_tensor(device_ptr: int, nbytes: int) -> torch.Tensor:
    import cupy as cp

    mem = cp.cuda.UnownedMemory(device_ptr, nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    arr = cp.ndarray((nbytes,), dtype=cp.uint8, memptr=memptr)
    return torch.as_tensor(arr, device="cuda")

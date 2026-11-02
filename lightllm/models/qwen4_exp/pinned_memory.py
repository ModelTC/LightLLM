import math
import mmap

import numpy as np
import torch

_CHUNK_BYTES = 256 * 1024 * 1024
_LARGE_BYTES = 512 * 1024 * 1024


class _RegisteredArray(np.ndarray):
    def __array_finalize__(self, source):
        self.allocation = getattr(source, "allocation", None)


class _Registration:
    def __init__(self, mapping, device, runtime):
        self.mapping = mapping
        self.device = device
        self.runtime = runtime
        self.addresses = []

    def __del__(self):
        if not self.addresses:
            return

        try:
            with torch.cuda.device(self.device):
                torch.cuda.synchronize()
        except Exception:
            pass
        for address in reversed(self.addresses):
            try:
                self.runtime.cudaHostUnregister(address)
            except Exception:
                pass


def pinned_zeros(shape, *, dtype, device="cpu", pin_memory=True):
    size = math.prod(shape) * dtype.itemsize
    if device != "cpu" or not pin_memory:
        raise ValueError("Qwen recurrent snapshots require pinned CPU storage")
    if size < _LARGE_BYTES:
        return torch.zeros(shape, dtype=dtype, device="cpu", pin_memory=True)

    cell_bytes = math.prod(shape[1:]) * dtype.itemsize
    registration_unit = math.lcm(cell_bytes, mmap.PAGESIZE)
    if registration_unit > _CHUNK_BYTES:
        raise ValueError("A recurrent snapshot must fit within one bounded host registration")
    chunk_bytes = (_CHUNK_BYTES // registration_unit) * registration_unit
    runtime = torch.cuda.cudart()
    mapping = mmap.mmap(-1, size)
    allocation = _Registration(mapping, torch.cuda.current_device(), runtime)
    array = np.frombuffer(mapping, dtype=np.uint8).view(_RegisteredArray)
    array.allocation = allocation
    value = torch.from_numpy(array).view(dtype).reshape(shape)
    value.zero_()
    for offset in range(0, size, chunk_bytes):
        address = value.data_ptr() + offset
        status = runtime.cudaHostRegister(address, min(chunk_bytes, size - offset), 0)
        if int(status):
            raise RuntimeError(
                f"Cannot register Qwen pinned snapshot at byte {offset}: {runtime.cudaGetErrorString(status)}"
            )
        allocation.addresses.append(address)

    value._qwen4_pinned_allocation = allocation
    return value

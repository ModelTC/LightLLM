"""Register existing host memory for direct access from CUDA kernels."""

from weakref import finalize

import torch


# Portable applies across CUDA contexts within this process. Each process registers its own mapping.
CUDA_HOST_REGISTER_PORTABLE = 1
CUDA_HOST_REGISTER_MAPPED = 2


def register_host_memory(address: int, size_bytes: int) -> None:
    """Register an existing host address range in the current process."""
    flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED
    torch.cuda.check_error(torch.cuda.cudart().cudaHostRegister(address, size_bytes, flags))


def unregister_host_memory(address: int) -> None:
    """Unregister the host range starting at the registered address."""
    torch.cuda.check_error(torch.cuda.cudart().cudaHostUnregister(address))


def register_cpu_tensor_storage(tensor: torch.Tensor) -> None:
    """Pin the full backing storage in place and keep it registered while any local view is alive."""
    if tensor.device.type != "cpu" or tensor.numel() == 0 or tensor.is_pinned():
        return

    storage = tensor.untyped_storage()
    address = storage.data_ptr()
    device_id = torch.cuda.current_device()
    register_host_memory(address, storage.nbytes())
    cleanup = finalize(storage, _unregister_cpu_storage, address, device_id)
    # Process teardown releases CUDA registrations after any remaining tensors.
    cleanup.atexit = False


def _unregister_cpu_storage(address: int, device_id: int) -> None:
    torch.cuda.synchronize(device=device_id)
    unregister_host_memory(address)

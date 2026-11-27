import threading

import torch

from lightllm.utils.dist_utils import get_current_device_id

# (device, name, dtype) -> requested size -> buffer; sizes of one name share the largest storage.
_workspace_buffers: dict[tuple, dict[int, torch.Tensor]] = {}
_workspace_buffer_lock = threading.Lock()


def get_gpu_workspace_buffer(key_name: str, workspace_size: int, dtype: torch.dtype = torch.int8) -> torch.Tensor:
    """Return a process-local workspace of workspace_size elements of dtype on the current device.

    Requests with one key name and dtype share storage: a request no larger than a buffer the name
    already has gets a prefix view of it, so a draft model reuses the main model's reservation instead
    of allocating its own. A name therefore stands for one user at a time; buffers that are live at the
    same time (two microbatches, two kernels in flight on different streams) need different names.
    Repeated requests return the same tensor, so CUDA graphs keep reading one address.
    """
    if not key_name:
        raise ValueError("workspace key_name must not be empty")
    if workspace_size <= 0:
        raise ValueError(f"workspace_size must be positive, got {workspace_size}")

    device_id = get_current_device_id()
    with _workspace_buffer_lock:
        buffers = _workspace_buffers.setdefault((device_id, key_name, dtype), {})
        buffer = buffers.get(workspace_size)
        if buffer is None:
            larger = [buffers[size] for size in sorted(buffers) if size > workspace_size]
            if larger:
                buffer = larger[0][:workspace_size]
            else:
                buffer = torch.empty(workspace_size, dtype=dtype, device=device_id)
            buffers[workspace_size] = buffer
        return buffer

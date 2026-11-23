import threading

import torch

from lightllm.utils.dist_utils import get_current_device_id

_workspace_buffers: dict[tuple, torch.Tensor] = {}
_workspace_buffer_lock = threading.Lock()


def get_gpu_workspace_buffer(key_name: str, workspace_size: int, dtype: torch.dtype = torch.int8) -> torch.Tensor:
    """Return a process-local workspace of workspace_size elements, shared by key name, size, dtype and device."""
    if not key_name:
        raise ValueError("workspace key_name must not be empty")
    if workspace_size <= 0:
        raise ValueError(f"workspace_size must be positive, got {workspace_size}")

    device_id = get_current_device_id()
    buffer_key = (device_id, key_name, workspace_size, dtype)
    with _workspace_buffer_lock:
        workspace_buffer = _workspace_buffers.get(buffer_key)
        if workspace_buffer is None:
            workspace_buffer = torch.empty(workspace_size, dtype=dtype, device=device_id)
            _workspace_buffers[buffer_key] = workspace_buffer
        return workspace_buffer

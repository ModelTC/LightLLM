"""CUDA IPC transport for the request-to-token page table.

This module deliberately has no dependency on ``ReqManager`` or ``basemodel``.
Generation workers only need the shared page-table tensor; importing the full
request allocator would couple their process startup to inference internals.
"""

from multiprocessing.reduction import ForkingPickler
from typing import TYPE_CHECKING

from filelock import FileLock

from lightllm.utils.dist_utils import get_current_rank_in_node, get_node_world_size
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_utils import create_or_link_shm

if TYPE_CHECKING:
    import torch


_HEADER_SIZE = 8


def write_req_to_token_indexs_to_shm(req_to_token_indexs: "torch.Tensor") -> None:
    """Publish one rank-local request-to-token tensor for colocated readers."""

    copy_count = get_node_world_size() * 2
    # A multiprocessing reduction token can be detached only once.  Serialize
    # every reader copy independently instead of duplicating the same bytes.
    tensor_bytes_array = [
        ForkingPickler.dumps(req_to_token_indexs).tobytes() for _ in range(copy_count)
    ]
    tensor_size = len(tensor_bytes_array[0])
    if any(len(tensor_bytes) != tensor_size for tensor_bytes in tensor_bytes_array):
        raise RuntimeError("request-to-token IPC payload sizes differ")

    server_name = get_unique_server_name()
    shm_name = f"{server_name}_req_to_token_{get_current_rank_in_node()}"
    lock = FileLock(f"/tmp/{server_name}_req_to_token_lock")
    with lock:
        shm = create_or_link_shm(
            name=shm_name,
            expected_size=tensor_size * copy_count + _HEADER_SIZE,
            force_mode="create",
        )
        try:
            shm.buf[0:4] = copy_count.to_bytes(4, "little")
            shm.buf[4:8] = tensor_size.to_bytes(4, "little")
            for copy_index, tensor_bytes in enumerate(tensor_bytes_array):
                start = _HEADER_SIZE + copy_index * tensor_size
                shm.buf[start : start + tensor_size] = tensor_bytes
        finally:
            shm.close()


def load_req_to_token_indexs_from_shm(rank_in_node: int) -> "torch.Tensor":
    """Attach to one target rank's request-to-token tensor through CUDA IPC."""

    server_name = get_unique_server_name()
    shm_name = f"{server_name}_req_to_token_{rank_in_node}"
    lock = FileLock(f"/tmp/{server_name}_req_to_token_lock")
    with lock:
        shm = create_or_link_shm(
            name=shm_name,
            expected_size=-1,
            force_mode="link",
        )
        try:
            left_count = int.from_bytes(shm.buf[0:4], "little")
            tensor_size = int.from_bytes(shm.buf[4:8], "little")
            if left_count <= 0:
                raise RuntimeError(
                    f"request-to-token IPC copies exhausted for rank {rank_in_node}"
                )
            start = _HEADER_SIZE + (left_count - 1) * tensor_size
            tensor_bytes = shm.buf[start : start + tensor_size].tobytes()
            shm.buf[0:4] = (left_count - 1).to_bytes(4, "little")
        finally:
            shm.close()
    return ForkingPickler.loads(tensor_bytes)

import os
import uuid
from multiprocessing import shared_memory

import pytest
import torch

from lightllm.common import req_to_token_ipc


def test_req_to_token_ipc_round_trip_and_copy_budget(monkeypatch):
    server_name = f"test_req_to_token_{os.getpid()}_{uuid.uuid4().hex}"
    shm_name = f"{server_name}_req_to_token_0"
    monkeypatch.setattr(req_to_token_ipc, "get_unique_server_name", lambda: server_name)
    monkeypatch.setattr(req_to_token_ipc, "get_current_rank_in_node", lambda: 0)
    monkeypatch.setattr(req_to_token_ipc, "get_node_world_size", lambda: 1)

    expected = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    try:
        req_to_token_ipc.write_req_to_token_indexs_to_shm(expected)

        first = req_to_token_ipc.load_req_to_token_indexs_from_shm(0)
        second = req_to_token_ipc.load_req_to_token_indexs_from_shm(0)
        assert torch.equal(first, expected)
        assert torch.equal(second, expected)
        with pytest.raises(RuntimeError, match="IPC copies exhausted"):
            req_to_token_ipc.load_req_to_token_indexs_from_shm(0)
    finally:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
        except FileNotFoundError:
            pass
        else:
            shm.close()
            shm.unlink()

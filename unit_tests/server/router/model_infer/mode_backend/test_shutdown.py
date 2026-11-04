import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lightllm.server.router.model_infer.mode_backend import base_backend
from lightllm.server.router.model_infer.mode_backend.chunked_prefill import impl as chunked
from lightllm.server.router.model_infer.mode_backend.dp_backend import impl as dp
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventManager


@pytest.mark.parametrize(
    "module,backend_type", [(chunked, chunked.ChunkedPrefillBackend), (dp, dp.DPChunkedPrefillBackend)]
)
@pytest.mark.parametrize("overlap", [False, True])
def test_shutdown_releases_both_inference_threads(module, backend_type, overlap, monkeypatch):
    monkeypatch.setattr(module, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    backend = backend_type.__new__(backend_type)
    backend.overlap_event_manager = OverlapEventManager()
    backend._inference_stopped = False
    backend.support_overlap = overlap
    backend.classed_req_no_decode = False
    backend.classed_req_strict_prefill = True
    backend.logger = Mock()
    visited, finished, failures = [], [], []
    backend._try_read_new_reqs = lambda: len(visited) >= 4
    backend._get_classed_reqs = lambda **kwargs: ([], [], 0)
    backend._dp_all_gather_prefill_and_decode_req_num = lambda **kwargs: (
        (None, None) if kwargs["stopping"] else (np.array([0, 0]), np.array([0, 0]))
    )

    def select(**kwargs):
        visited.append(threading.get_ident())
        return SimpleNamespace(is_prefill=lambda: False, is_decode=lambda: False, is_pass=lambda: True)

    backend.control_state_machine = SimpleNamespace(try_recover_paused_reqs=lambda: False, select_run_way=select)

    def run():
        try:
            backend.infer_loop()
            finished.append(threading.get_ident())
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=run, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    assert len(visited) == 4 and len(set(visited)) == 2
    assert len(set(finished)) == 2


@pytest.mark.parametrize("master", [False, True])
def test_node_stop_is_broadcast_before_reading_requests(master, monkeypatch):
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend.is_master_in_node = master
    backend._shutdown_requested = threading.Event()
    backend._shutdown_requested.set()
    backend.args = SimpleNamespace(node_rank=1)
    backend.node_world_size = 2
    backend.node_nccl_group = "node"
    backend.node_broadcast_tensor = torch.zeros(1, dtype=torch.int32)
    backend.shm_reqs_io_buffer = SimpleNamespace(is_ready=lambda: pytest.fail("must not read requests after stop"))

    def broadcast(tensor, **kwargs):
        assert kwargs == dict(src=2, group="node", async_op=False)
        if master:
            assert tensor.item() == -1
        else:
            tensor.fill_(-1)

    monkeypatch.setattr(base_backend, "broadcast", broadcast)
    assert backend._try_read_new_reqs_normal()


def test_multinode_stop_propagates_from_a_remote_rank(monkeypatch):
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend._shutdown_requested = threading.Event()
    backend.shm_reqs_io_buffer = SimpleNamespace(is_ready=lambda: True)
    backend.multinode_tp_gather_item_tensor = torch.zeros(1, dtype=torch.int32)
    backend.multinode_tp_all_gather_tensor = torch.zeros(4, dtype=torch.int32)
    backend.multinode_tp_nccl_group = "tp"

    def gather(output, value, **kwargs):
        assert value.item() == 1
        assert kwargs == dict(group="tp", async_op=False)
        output.copy_(torch.tensor([1, 1, -1, 1]))

    monkeypatch.setattr(base_backend, "all_gather_into_tensor", gather)
    assert backend._try_read_new_reqs_multinode_tp()


@pytest.mark.parametrize("local_stop", [False, True])
def test_dp_stop_reaches_all_ranks_before_decode_count_collective(local_stop, monkeypatch):
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend.dp_gather_item_tensor = torch.zeros(1, dtype=torch.int32)
    backend.dp_all_gather_tensor = torch.zeros(4, dtype=torch.int32)
    calls = []

    def gather(output, value, **kwargs):
        calls.append(value.item())
        output.copy_(torch.tensor([-1, -1, 2, 2]))

    monkeypatch.setattr(base_backend, "all_gather_into_tensor", gather)
    assert backend._dp_all_gather_prefill_and_decode_req_num([None, None], [None], local_stop) == (None, None)
    assert calls == ([-1] if local_stop else [2])


@pytest.mark.parametrize("sync_failure", [False, True])
def test_shutdown_retains_graphs_and_groups_until_execution_has_finished(sync_failure, monkeypatch):
    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend._shutdown_requested = threading.Event()
    calls = []

    def join(index):
        assert backend._shutdown_requested.is_set()
        calls.append(index)

    backend.infer_loop_thread = SimpleNamespace(join=lambda: join(0))
    backend.infer_loop_thread1 = SimpleNamespace(join=lambda: join(1))
    backend.model = SimpleNamespace(graph_runner=SimpleNamespace(close=lambda: calls.append("main")))
    backend.draft_models = [SimpleNamespace(graph_runner=SimpleNamespace(close=lambda: calls.append("draft")))]
    backend.logger = Mock()

    def sync():
        calls.append("sync")
        if sync_failure:
            raise RuntimeError("in-flight work failed")

    backend.execution_device = SimpleNamespace(synchronize=sync)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: pytest.fail("default CUDA synchronization"))
    monkeypatch.setattr(base_backend.dist_group_manager, "close", lambda: calls.append("buffers"))
    monkeypatch.setattr(base_backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(base_backend.dist, "destroy_process_group", lambda: calls.append("groups"))
    if sync_failure:
        with pytest.raises(RuntimeError, match="in-flight work failed"):
            backend.shutdown()
        assert calls == [0, 1, "sync"]
    else:
        backend.shutdown()
        assert calls == [0, 1, "sync", "main", "draft", "buffers", "groups"]

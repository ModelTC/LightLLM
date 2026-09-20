from concurrent.futures import ThreadPoolExecutor
from multiprocessing import resource_tracker, shared_memory
from threading import Event

import pytest

from lightllm.utils import shm_utils


def test_get_service_shm_name_requires_service_name(monkeypatch):
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: None)

    with pytest.raises(RuntimeError, match="LIGHTLLM_UNIQUE_SERVICE_NAME_ID is unset"):
        shm_utils.get_service_shm_name("req_pool")


def test_get_service_shm_name_adds_prefix_once(monkeypatch):
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: "service_uuid_0")

    assert shm_utils.get_service_shm_name("req_pool") == "service_uuid_0_req_pool"
    assert shm_utils.get_service_shm_name("service_uuid_0_req_pool") == "service_uuid_0_req_pool"


def test_create_or_link_shm_passes_scoped_name_to_shared_memory_layer(monkeypatch):
    created_names = []
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: "service_uuid_1")
    monkeypatch.setattr(
        shm_utils,
        "_force_create_shm",
        lambda name, expected_size: created_names.append((name, expected_size)) or object(),
    )

    shm_utils.create_or_link_shm("token_load", 128, force_mode="create")

    assert created_names == [("service_uuid_1_token_load", 128)]


@pytest.mark.parametrize("method, tracker_method", [("__init__", "register"), ("unlink", "unregister")])
def test_service_shm_concurrent_calls_restore_tracker(monkeypatch, method, tracker_method):
    original_tracker_method = getattr(resource_tracker, tracker_method)
    # 即使回归导致 patch 恢复错误，也不污染后续测试。
    monkeypatch.setattr(resource_tracker, tracker_method, original_tracker_method)
    first_entered = Event()
    second_started = Event()
    second_entered = Event()
    release_first = Event()
    first_finished = Event()

    def ordered_operation(self, *args, **kwargs):
        if self._test_index == 0:
            first_entered.set()
            assert release_first.wait(5)
        else:
            second_entered.set()
            # 无锁时强制先退出第一个 patch，再退出第二个，复现恢复顺序错误。
            assert first_finished.wait(5)

    monkeypatch.setattr(shared_memory.SharedMemory, method, ordered_operation)

    def worker(index):
        # 只测试 patch 生命周期，不申请真实共享内存。
        shm = shm_utils.ServiceSharedMemory.__new__(shm_utils.ServiceSharedMemory)
        shm._test_index = index
        if index == 1:
            second_started.set()
        try:
            if method == "__init__":
                shm.__init__(name="test_tracker")
            else:
                shm.unlink()
        finally:
            if index == 0:
                first_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(worker, 0)
        try:
            assert first_entered.wait(5)
            second = executor.submit(worker, 1)
            assert second_started.wait(5)
            overlapped = second_entered.wait(0.2)
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert not overlapped, "Concurrent calls entered the patched operation together"
    assert getattr(resource_tracker, tracker_method) is original_tracker_method

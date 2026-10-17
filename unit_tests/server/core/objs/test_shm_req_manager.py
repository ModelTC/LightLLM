import os
import asyncio
import pytest
import time
import uuid
import numpy as np
from unittest.mock import MagicMock
from types import SimpleNamespace

from easydict import EasyDict
from lightllm.utils.envs_utils import set_env_start_args, get_env_start_args
from lightllm.utils import shm_utils
from lightllm.server.core.objs.shm_req_manager import ShmReqManager


@pytest.fixture(scope="module", autouse=True)
def setup_env():
    monkeypatch = pytest.MonkeyPatch()
    service_name = "test_shm_req_manager_" + uuid.uuid4().hex
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: service_name)
    original = os.environ.get("LIGHTLLM_START_ARGS")
    set_env_start_args(
        EasyDict(
            running_max_req_size=10,
            disable_chunked_prefill=True,
            mtp_step=0,
            llm_prefill_att_backend=["None"],
            llm_decode_att_backend=["None"],
            cpu_cache_token_page_size=256,
            enable_cpu_cache=False,
        )
    )
    # clear the lru_cache if used
    if hasattr(get_env_start_args, "cache_clear"):
        get_env_start_args.cache_clear()

    yield
    if original is not None:
        os.environ["LIGHTLLM_START_ARGS"] = original
    else:
        os.environ.pop("LIGHTLLM_START_ARGS", None)
    if hasattr(get_env_start_args, "cache_clear"):
        get_env_start_args.cache_clear()
    monkeypatch.undo()


@pytest.fixture(scope="module")
def shm_req_manager():
    return ShmReqManager()


def test_init_reqs_shm(shm_req_manager):
    assert shm_req_manager.reqs_shm is not None
    assert shm_req_manager.reqs_shm.size == shm_req_manager.req_shm_byte_size


def test_alloc_req_index(shm_req_manager):
    index = shm_req_manager.alloc_req_index()
    assert index is not None
    assert shm_req_manager.alloc_state_shm.arr[index] == 1


def test_release_req_index(shm_req_manager):
    index = shm_req_manager.alloc_req_index()
    assert index is not None
    shm_req_manager.release_req_index(index)
    assert shm_req_manager.alloc_state_shm.arr[index] == 0


def test_get_req_obj_by_index(shm_req_manager):
    index = shm_req_manager.alloc_req_index()
    req_obj = shm_req_manager.get_req_obj_by_index(index)
    assert req_obj is not None
    assert req_obj.ref_count == 1


def test_put_back_req_obj(shm_req_manager):
    index = shm_req_manager.alloc_req_index()
    req_obj = shm_req_manager.get_req_obj_by_index(index)
    prompt_ids = req_obj.shm_prompt_ids = MagicMock()
    logprobs = req_obj.shm_logprobs = MagicMock()

    shm_req_manager.put_back_req_obj(req_obj)

    assert req_obj.ref_count == 0
    prompt_ids.detach_shm.assert_called_once_with()
    logprobs.detach_shm.assert_called_once_with()
    assert not hasattr(req_obj, "shm_prompt_ids")
    assert not hasattr(req_obj, "shm_logprobs")
    shm_req_manager.release_req_index(index)


def test_alloc_req_index_no_available(shm_req_manager):
    for _ in range(shm_req_manager.max_req_num):
        shm_req_manager.alloc_req_index()
    index = shm_req_manager.alloc_req_index()
    assert index is None  # No indices should be available


def test_performance_alloc_release(shm_req_manager):
    start_time = time.time()

    for _ in range(100):
        index = shm_req_manager.alloc_req_index()
        if index is not None:
            shm_req_manager.release_req_index(index)

    duration = time.time() - start_time
    print(f"Time taken for 100 alloc/release cycles: {duration:.6f} seconds")


@pytest.fixture
def batch_req_manager(monkeypatch):
    service_name = f"test_batch_req_manager_{uuid.uuid4().hex}"
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: service_name)
    manager = ShmReqManager()
    yield manager
    manager.reqs = None
    manager.linked_req_manager._values = None
    manager.linked_req_manager._shm_array.close_shm()
    manager.alloc_state_shm.close_shm()
    for shm in (manager.reqs_shm, manager.reqs_lock.shm, manager.manager_lock.shm):
        shm.close()
        shm.unlink()


def test_batch_allocation_and_reuse(batch_req_manager):
    indexes = batch_req_manager.alloc_req_indexes(3)
    assert len(set(indexes)) == 3
    assert np.all(batch_req_manager.alloc_state_shm.arr[indexes] == 1)
    batch_req_manager.release_req_index(indexes[0])
    remaining = batch_req_manager.alloc_req_indexes(batch_req_manager.max_req_num - 2)
    assert len(set(remaining)) == batch_req_manager.max_req_num - 2
    assert indexes[0] in remaining
    assert set(indexes[1:]).isdisjoint(remaining)
    assert batch_req_manager.alloc_req_index() is None


def test_insufficient_batch_does_not_change_allocation(batch_req_manager):
    batch_req_manager.alloc_req_indexes(batch_req_manager.max_req_num - 1)
    allocation = batch_req_manager.alloc_state_shm.arr.copy()
    linked_list = batch_req_manager.linked_req_manager._values.copy()
    assert batch_req_manager.alloc_req_indexes(2) is None
    np.testing.assert_array_equal(batch_req_manager.alloc_state_shm.arr, allocation)
    np.testing.assert_array_equal(batch_req_manager.linked_req_manager._values, linked_list)
    assert batch_req_manager.alloc_req_index() is not None


def test_batch_larger_than_capacity(batch_req_manager):
    assert batch_req_manager.alloc_req_indexes(batch_req_manager.max_req_num + 1) is None
    assert batch_req_manager.is_idle()
    assert len(batch_req_manager.alloc_req_indexes(batch_req_manager.max_req_num)) == batch_req_manager.max_req_num


@pytest.mark.parametrize("req_num", [0, -1])
def test_batch_requires_positive_count(batch_req_manager, req_num):
    with pytest.raises(ValueError, match="req_num must be positive"):
        batch_req_manager.alloc_req_indexes(req_num)
    assert batch_req_manager.is_idle()


def test_async_batch_allocation(batch_req_manager):
    async def allocate():
        indexes = await batch_req_manager.async_alloc_req_indexes(batch_req_manager.max_req_num)
        assert len(indexes) == batch_req_manager.max_req_num
        assert await batch_req_manager.async_alloc_req_indexes(1) is None
        await batch_req_manager.async_release_req_index(indexes[0])
        assert await batch_req_manager.async_alloc_req_indexes(1) == [indexes[0]]

    asyncio.run(allocate())


def test_concurrent_batches_do_not_hold_partial_slots(batch_req_manager):
    async def run():
        results = await asyncio.gather(*(batch_req_manager.async_alloc_req_indexes(3) for _ in range(16)))
        allocated = [index for indexes in results if indexes is not None for index in indexes]
        assert len(allocated) == len(set(allocated)) == 9
        assert batch_req_manager.alloc_req_index() is not None
        assert batch_req_manager.alloc_req_index() is None

    asyncio.run(run())


def test_http_batch_requests_make_progress_with_two_available_slots(batch_req_manager):
    from lightllm.server.httpserver.manager import HttpServerManager

    async def run():
        manager = HttpServerManager.__new__(HttpServerManager)
        manager.args = SimpleNamespace(running_max_req_size=batch_req_manager.max_req_num, run_mode="normal")
        manager.is_multinode_tp_slave = False
        manager.shm_req_manager = batch_req_manager
        occupied = batch_req_manager.alloc_req_indexes(batch_req_manager.max_req_num - 1)

        first_task = asyncio.create_task(manager._alloc_shm_req_indexes(2))
        await asyncio.sleep(0)
        # 只剩一个槽时，第一个双结果请求等待，但不会先占走这个槽。
        assert np.count_nonzero(batch_req_manager.alloc_state_shm.arr == 0) == 1
        assert not first_task.done()
        batch_req_manager.release_req_index(occupied.pop())

        # 第二个请求可以一次拿到两个槽，释放后第一个请求也能继续，避免原先的互等。
        second_indexes = await asyncio.wait_for(manager._alloc_shm_req_indexes(2), timeout=2)
        for index in second_indexes:
            batch_req_manager.release_req_index(index)
        first_indexes = await asyncio.wait_for(first_task, timeout=2)
        assert len(first_indexes) == 2
        for index in occupied + first_indexes:
            batch_req_manager.release_req_index(index)
        assert batch_req_manager.is_idle()

    asyncio.run(run())


if __name__ == "__main__":
    pytest.main()

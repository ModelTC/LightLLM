import os
import pytest
import time
from easydict import EasyDict
from lightllm.utils.envs_utils import set_env_start_args, get_env_start_args
from lightllm.server.core.objs.shm_req_manager import ShmReqManager


@pytest.fixture(scope="module", autouse=True)
def setup_env():
    original = os.environ.get("LIGHTLLM_START_ARGS")
    set_env_start_args(
        EasyDict(
            running_max_req_size=10,
            disable_chunked_prefill=True,
            token_healing_mode=False,
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
    shm_req_manager.put_back_req_obj(req_obj)
    assert req_obj.ref_count == 0
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


if __name__ == "__main__":
    pytest.main()

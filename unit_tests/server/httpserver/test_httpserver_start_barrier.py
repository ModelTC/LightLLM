import os
import shutil
import time
import uuid
from multiprocessing import get_context
from pathlib import Path

from lightllm.server.httpserver.manager import wait_httpserver_workers
from lightllm.utils.envs_utils import get_unique_server_name


def _barrier_worker(service_name: str, delay: float, result_queue) -> None:
    os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = service_name
    get_unique_server_name.cache_clear()
    time.sleep(delay)
    arrived = time.time()
    wait_httpserver_workers(worker_num=2)
    result_queue.put((arrived, time.time()))


def test_one_httpserver_does_not_wait():
    started = time.time()
    wait_httpserver_workers(worker_num=1)
    assert time.time() - started < 0.1


def test_httpservers_wait_until_all_have_started():
    service_name = f"test_httpserver_barrier_{uuid.uuid4().hex}"
    previous = os.environ.get("LIGHTLLM_UNIQUE_SERVICE_NAME_ID")
    context = get_context("fork")
    result_queue = context.Queue()
    processes = [
        context.Process(target=_barrier_worker, args=(service_name, delay, result_queue)) for delay in (0.0, 0.3)
    ]
    try:
        for process in processes:
            process.start()
        results = sorted(result_queue.get(timeout=5) for _ in processes)
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
        early_done = results[0][1]
        late_arrived = results[1][0]
        assert early_done >= late_arrived - 0.05
        assert not Path(f"/tmp/{service_name}_httpserver_start").exists()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        shutil.rmtree(f"/tmp/{service_name}_httpserver_start", ignore_errors=True)
        Path(f"/tmp/{service_name}_httpserver_start.lock").unlink(missing_ok=True)
        if previous is None:
            os.environ.pop("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", None)
        else:
            os.environ["LIGHTLLM_UNIQUE_SERVICE_NAME_ID"] = previous
        get_unique_server_name.cache_clear()

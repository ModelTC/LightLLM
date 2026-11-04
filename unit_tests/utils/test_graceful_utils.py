import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

from lightllm.utils.graceful_utils import graceful_registry


def _resource_owner(ready):
    graceful_registry("test_resource_owner")
    queue = mp.Queue()
    lock = mp.RLock()
    ready.send([item._semlock.name for item in (queue._rlock, queue._wlock, queue._sem, lock)])
    ready.close()
    while True:
        time.sleep(0.1)


def test_launcher_exit_signal_runs_worker_resource_finalizers():
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    worker = context.Process(target=_resource_owner, args=(writer,))
    worker.start()
    writer.close()
    try:
        assert reader.poll(20), "worker did not publish its resources"
        names = reader.recv()
        paths = [Path("/dev/shm") / ("sem." + name.lstrip("/")) for name in names]
        assert all(path.exists() for path in paths)
        os.kill(worker.pid, signal.SIGTERM)
        worker.join(timeout=0.2)
        assert worker.is_alive(), "external SIGTERM must wait for the launcher"
        os.kill(worker.pid, signal.SIGUSR2)
        worker.join(timeout=10)
        assert worker.exitcode == 0
        assert all(not path.exists() for path in paths)
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=10)
        reader.close()
        worker.close()

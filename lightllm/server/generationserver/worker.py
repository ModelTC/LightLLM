"""Process-safe lifecycle shared by model-owned generation workers."""

import asyncio
import os
import time
import uuid

import rpyc
import torch.multiprocessing as mp
from rpyc.utils.classic import obtain
from rpyc.utils.factory import unix_connect

from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.retry_utils import retry
from lightllm.utils.torch_memory_saver_utils import TorchMemorySaverWrapper


logger = init_logger(__name__)
_WORKER_JOIN_TIMEOUT_SECONDS = 5.0
_WORKER_START_TIMEOUT_SECONDS = 40.0
_WORKER_START_POLL_SECONDS = 0.1


def close_worker_resources(rpc_conn, process, socket_path):
    errors = []
    if rpc_conn is not None:
        try:
            rpc_conn.close()
        except Exception as exc:
            errors.append(exc)

    if process is not None and process.pid is not None:
        try:
            if process.is_alive():
                process.terminate()
            process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
            if process.is_alive():
                process.kill()
                process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
        except Exception as exc:
            errors.append(exc)

    if socket_path is not None:
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            errors.append(exc)

    if errors:
        raise errors[0]


class ModelRpcClient:
    def __init__(self, rpc_conn, process=None, socket_path=None):
        self.rpc_conn = rpc_conn
        self.process = process
        self.socket_path = socket_path
        self._closed = False
        self._close_lock = asyncio.Lock()

        def async_wrap(func):
            remote = rpyc.async_(func)

            async def wrapped(*args, **kwargs):
                result = remote(*args, **kwargs)
                await asyncio.to_thread(result.wait)
                return obtain(result.value)

            return wrapped

        self._init_model = async_wrap(self.rpc_conn.root.init_model)
        self._run_task = async_wrap(self.rpc_conn.root.run_task)

    async def init_model(self, kvargs):
        return await self._init_model(kvargs)

    async def run_task(self, task):
        return await self._run_task(task)

    async def close(self):
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            await asyncio.to_thread(
                close_worker_resources,
                self.rpc_conn,
                self.process,
                self.socket_path,
            )


def wait_for_model_process(success_event, process, timeout, *, worker_name):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"{worker_name} worker did not initialize within {timeout:g}s"
            )
        if success_event.wait(timeout=min(_WORKER_START_POLL_SECONDS, remaining)):
            if not process.is_alive():
                raise RuntimeError(
                    f"{worker_name} worker exited during initialization "
                    f"with code {process.exitcode}"
                )
            return
        if not process.is_alive():
            raise RuntimeError(
                f"{worker_name} worker exited before initialization "
                f"with code {process.exitcode}"
            )


async def start_model_worker(init_env, *, socket_prefix: str, worker_name: str):
    socket_path = f"/tmp/{socket_prefix}_{uuid.uuid4().hex[:8]}.sock"
    success_event = mp.Event()
    process = mp.Process(target=init_env, args=(socket_path, success_event))
    connection = None
    try:
        torch_memory_saver = TorchMemorySaverWrapper(
            get_env_start_args().enable_torch_memory_saver
        )
        with torch_memory_saver.configure_subprocess():
            process.start()
        await asyncio.to_thread(
            wait_for_model_process,
            success_event,
            process,
            _WORKER_START_TIMEOUT_SECONDS,
            worker_name=worker_name,
        )
        connection = await asyncio.to_thread(
            retry(max_attempts=20, wait_time=2)(unix_connect),
            socket_path,
            config={"allow_pickle": True},
        )
        if not process.is_alive():
            raise RuntimeError(
                f"{worker_name} worker exited while establishing its RPC "
                f"connection with code {process.exitcode}"
            )
        return ModelRpcClient(connection, process, socket_path)
    except BaseException:
        try:
            await asyncio.to_thread(
                close_worker_resources,
                connection,
                process,
                socket_path,
            )
        except Exception:
            logger.exception("failed to clean up generation model worker startup")
        raise

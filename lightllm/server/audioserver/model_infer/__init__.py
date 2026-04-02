import asyncio
import rpyc
import inspect
import uuid
import os
import multiprocessing
import setproctitle
from lightllm.utils.retry_utils import retry
from rpyc.utils.factory import unix_connect
from lightllm.utils.graceful_utils import graceful_registry
from .model_rpc_client import AudioModelRpcClient
from .model_rpc import AudioModelRpcServer
from ..objs import rpyc_config
from lightllm.utils.envs_utils import get_unique_server_name


def _init_env(socket_path: str, success_event):
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::audio_model_infer")

    import lightllm.utils.rpyc_fix_utils as _

    t = rpyc.ThreadedServer(AudioModelRpcServer(), socket_path=socket_path, protocol_config=rpyc_config)
    success_event.set()
    t.start()
    return


async def start_model_process():
    import lightllm.utils.rpyc_fix_utils as _

    socket_path = _generate_unix_socket_path()
    if os.path.exists(socket_path):
        os.remove(socket_path)

    success_event = multiprocessing.Event()
    proc = multiprocessing.Process(
        target=_init_env,
        args=(socket_path, success_event),
    )
    proc.start()
    await asyncio.to_thread(success_event.wait, timeout=40)
    assert proc.is_alive()

    conn = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config=rpyc_config)
    assert proc.is_alive()

    conn._bg_thread = rpyc.BgServingThread(conn, sleep_interval=0.001)

    return AudioModelRpcClient(conn)


def _generate_unix_socket_path() -> str:
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/lightllm_audio_model_infer_{unique_id}.sock"

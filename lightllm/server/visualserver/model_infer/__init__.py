import asyncio
import rpyc
import inspect
import uuid
import os
import multiprocessing
from lightllm.utils.retry_utils import retry
from rpyc.utils.factory import unix_connect
from rpyc.utils.classic import obtain
from rpyc.utils.server import ThreadedServer
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_env_start_args
from .model_rpc_client import VisualModelRpcClient
from .model_rpc import VisualModelRpcServer
from ..objs import rpyc_config


def _init_env(socket_path: str, success_event):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    import lightllm.utils.rpyc_fix_utils as _

    t = ThreadedServer(VisualModelRpcServer(), socket_path=socket_path, protocol_config=rpyc_config)
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
        args=(
            socket_path,
            success_event,
        ),
    )
    proc.start()
    await asyncio.to_thread(success_event.wait, timeout=40)
    assert proc.is_alive()

    conn = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config=rpyc_config)
    assert proc.is_alive()

    # 服务端需要调用客户端传入的event所以，客户端需要一个后台线程进行相关的处理。
    conn._bg_thread = rpyc.BgServingThread(conn, sleep_interval=0.001)

    return VisualModelRpcClient(conn)


def _generate_unix_socket_path() -> str:
    """Generate a random Unix socket path"""
    unique_id = uuid.uuid4().hex[:8]
    return f"/tmp/lightllm_model_infer_{unique_id}.sock"

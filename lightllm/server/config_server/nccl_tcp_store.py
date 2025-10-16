import psutil
import time
import setproctitle
import torch.distributed as dist
import torch.multiprocessing as mp
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name

logger = init_logger(__name__)


def start_tcp_store_server(nccl_store_host, nccl_store_port):
    """
    start a process to run a TCPStore server.
    """
    process = mp.Process(
        target=_start_tcp_store_server,
        args=(nccl_store_host, nccl_store_port),
        daemon=True,
    )
    process.start()
    return process


def _start_tcp_store_server(nccl_store_host, nccl_store_port):
    """
    start a TCPStore server.
    """
    start_parent_check_thread()
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::nccl_tcp_store")

    try:
        from torch._C._distributed_c10d import _DEFAULT_PG_NCCL_TIMEOUT

        default_pg_nccl_timeout = _DEFAULT_PG_NCCL_TIMEOUT
    except ImportError:
        # if C++ NCCL support is not compiled, we don't have access to the default nccl value.
        # if anyone is actually trying to use nccl in this state, it should error.
        default_pg_nccl_timeout = None

    logger.info(f"default_pg_nccl_timeout: {default_pg_nccl_timeout}")
    logger.info(f"[Server] TCPStore start: {nccl_store_host}:{nccl_store_port}")
    try:
        store = dist.TCPStore(
            host_name=nccl_store_host,
            port=nccl_store_port,
            world_size=None,
            is_master=True,
            wait_for_workers=False,
            timeout=default_pg_nccl_timeout,
            multi_tenant=True,
            use_libuv=True,
        )

        while True:
            keys_num = store.num_keys()
            logger.info(f"[Server] TCPStore start: {nccl_store_host}:{nccl_store_port} keys num: {keys_num}")
            time.sleep(20)

    except Exception as e:
        logger.warning(str(e))
        logger.info(f"TCPStore server {nccl_store_host}:{nccl_store_port} start failed, retrying ...")

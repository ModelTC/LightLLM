import argparse
import os
import subprocess
import sys
import time
import setproctitle
import torch.distributed as dist
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import start_parent_check_thread
from lightllm.utils.envs_utils import get_unique_server_name

logger = init_logger(__name__)


def start_tcp_store_server(nccl_store_host, nccl_store_port):
    """
    start a process to run a TCPStore server.
    """
    # Hypercorn workers are multiprocessing daemons and cannot create another
    # multiprocessing.Process. An exec'ed Python child can own the TCPStore and
    # retain the same parent-death monitoring as the other service processes.
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lightllm.server.config_server.nccl_tcp_store",
            "--host",
            nccl_store_host,
            "--port",
            str(nccl_store_port),
            "--parent-pid",
            str(os.getpid()),
        ]
    )


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
        logger.exception(f"TCPStore server {nccl_store_host}:{nccl_store_port} start failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    # The parent may exit during Python/torch imports, before its monitor starts.
    if os.getppid() == args.parent_pid:
        _start_tcp_store_server(args.host, args.port)

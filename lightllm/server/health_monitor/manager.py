import sys
import uvloop
import asyncio
import setproctitle

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
import inspect
import requests
import time

from lightllm.utils.log_utils import init_logger
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_port_args import get_shm_port_args

logger = init_logger(__name__)

consecutive_failures = 0


def health_monitor(url, all_process_ids):
    global consecutive_failures
    try:
        all_processes_is_alive = all_is_alive(all_process_ids)
        if all_processes_is_alive:
            response = requests.get(url, timeout=60)
            if response.status_code == 200:
                if consecutive_failures:
                    logger.info("event=health.recovered previous_failures=%s", consecutive_failures)
                consecutive_failures = 0
            else:
                raise Exception(f"Health check failed with status code: {response.status_code}")
        else:
            raise Exception("not all processes is alive")
    except asyncio.TimeoutError:
        logger.warning("event=health.check_failed reason=timeout")
        consecutive_failures += 1
    except Exception as e:
        logger.warning("event=health.check_failed reason=%s", e)
        consecutive_failures += 1

    if consecutive_failures >= 3:
        logger.critical("event=health.termination consecutive_failures=%s", consecutive_failures)
        import signal

        # 先kill父进程的子进程
        for pid in all_process_ids[1:]:
            os.kill(pid, signal.SIGKILL)
            logger.debug(f"Killing process with PID: {pid}")

        # 杀父进程
        os.kill(all_process_ids[0], signal.SIGKILL)
        logger.debug(f"Killing process with PID: {all_process_ids[0]}")

        # 自 kill
        sys.exit(-1)
    return


def is_process_active(pid):
    import psutil

    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def all_is_alive(all_process_ids):
    for pid in all_process_ids:
        if not is_process_active(pid):
            logger.error(f"error, pid {pid} is not alive")
            return False
    return True


def get_all_cared_pids():
    all_process_ids = []
    import psutil

    all_process_ids.append(os.getppid())
    parent = psutil.Process(os.getppid())
    children = parent.children(recursive=True)
    for child in children:
        if child.pid != os.getpid():
            all_process_ids.append(child.pid)
    return all_process_ids


def start_health_check_process(args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::health_monitor")
    pipe_writer.send("init ok")

    all_process_ids = get_all_cared_pids()
    logger.debug(f"health monitor care process ids {all_process_ids}")

    host, port = args.host, get_shm_port_args().port
    url = f"http://{host}:{port}/health".format(host=host, port=port)
    interval_seconds = int(os.environ.get("HEALTH_CHECK_INTERVAL_SECONDS", 88))
    logger.info("Waiting for the server to start up.")
    logger.info(f"check interval seconds {interval_seconds}")
    time.sleep(6)
    logger.info("event=health.monitor_started interval_s=%s", interval_seconds)

    while True:
        health_monitor(url, all_process_ids)
        time.sleep(interval_seconds)
    return

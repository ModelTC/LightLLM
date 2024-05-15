import sys
import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
from lightllm.utils.log_utils import init_logger
import requests
import time

logger = init_logger(__name__)

consecutive_failures = 0


def health_monitor(url):
    global consecutive_failures
    try:
        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            logger.error(f"Health check failed with status code: {response.status_code}")
            consecutive_failures += 1
        else:
            logger.info("Health check passed")
            consecutive_failures = 0
    except asyncio.TimeoutError:
        logger.error("Health check request timed out")
        consecutive_failures += 1
    except Exception as e:
        logger.error(f"Health check failed with exception: {e}")
        consecutive_failures += 1

    if consecutive_failures >= 3:
        logger.error(f"kill all process consecutive_failures: {consecutive_failures}")
        import psutil
        import signal

        parent = psutil.Process(os.getppid())
        children = parent.children(recursive=True)
        for child in children:
            if child.pid != os.getpid():
                os.kill(child.pid, signal.SIGKILL)
                logger.debug(f"Killing process with PID: {child.pid}")
        os.kill(os.getppid(), signal.SIGKILL)
        logger.debug(f"Killing process with PID: {os.getppid()}")
        sys.exit(-1)
    return


def start_health_check_process(args, pipe_writer):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    pipe_writer.send("init ok")
    global consecutive_failures
    host, port = args.host, args.port
    url = f"http://{host}:{port}/health".format(host=host, port=port)
    interval_seconds = int(os.environ.get("HEALTH_CHECK_INTERVAL_SECONDS", 88))
    logger.info("Waiting for the server to start up.")
    logger.info(f"check interval seconds {interval_seconds}")
    time.sleep(6)

    logger.info("Server has started, starting the health check process.")
    while True:
        health_monitor(url)
        time.sleep(interval_seconds)
    return

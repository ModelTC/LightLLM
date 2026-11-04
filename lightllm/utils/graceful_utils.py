import sys
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def graceful_registry(sub_module_name):
    from lightllm.utils.log_utils import set_log_context

    set_log_context(component=sub_module_name)
    import signal

    # 子进程在受到 SIGTERM的时候，不能自己就提前退出。
    def graceful_shutdown(signum, frame):
        if signum == signal.SIGUSR2:
            # The launcher sends this only after the HTTP server has drained.
            # Returning through multiprocessing's bootstrap runs resource finalizers.
            logger.info(f"{sub_module_name} exiting after launcher shutdown request")
            sys.exit(0)
        logger.info(f"{sub_module_name} Received signal to shutdown. Performing graceful shutdown...")
        if signum == signal.SIGTERM:
            # 不退出，由主进程来决定退出时机
            logger.info(f"{sub_module_name} recive sigterm")

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGUSR2, graceful_shutdown)
    return

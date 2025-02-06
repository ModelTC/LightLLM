import os
import time
import threading
import psutil
import signal
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def is_process_active(pid):
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def kill_child_processes(parent_pid):
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        try:
            os.kill(child.pid, signal.SIGKILL)
            logger.warning(f"kill pid {child.pid}")
        except BaseException as e:
            logger.warning(f"kill pid {child.pid} failed {str(e)}")


def check_parent_alive():
    parent_pid = os.getppid()
    while True:
        if not is_process_active(parent_pid):
            logger.warning(f"parent is dead, kill self {os.getpid()}")
            kill_child_processes(os.getpid())
            os.kill(os.getpid(), signal.SIGKILL)

        time.sleep(10)
    return


def start_parent_check_thread():
    """
    检测父进程是否健康，如果出现问题，清理退出所有进程
    """
    thread = threading.Thread(target=check_parent_alive)
    thread.start()

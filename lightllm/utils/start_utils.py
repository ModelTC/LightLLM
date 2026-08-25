import os
import signal
import subprocess
import sys
import time
import multiprocessing as mp
import psutil
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import is_process_active

logger = init_logger(__name__)


class SubmoduleManager:
    def __init__(self):
        self.processes = []
        self.process_names = {}

    def start_submodule_processes(self, start_funcs=[], start_args=[]):
        assert len(start_funcs) == len(start_args)
        pipe_readers = []
        processes = []

        for start_func, start_arg in zip(start_funcs, start_args):
            pipe_reader, pipe_writer = mp.Pipe(duplex=False)
            process = mp.Process(
                target=start_func,
                args=start_arg + (pipe_writer,),
            )
            process.start()
            pipe_readers.append(pipe_reader)
            processes.append(process)

        # Wait for all processes to initialize
        for index, pipe_reader in enumerate(pipe_readers):
            init_state = pipe_reader.recv()
            if init_state != "init ok":
                logger.error(f"init func {start_funcs[index].__name__} : {str(init_state)}")
                for proc in processes:
                    proc.kill()
                sys.exit(1)
            else:
                logger.info(f"init func {start_funcs[index].__name__} : {str(init_state)}")

        assert all([proc.is_alive() for proc in processes])
        processes = [psutil.Process(proc.pid) for proc in processes]
        self.processes.extend(processes)
        self.process_names.update((process, process.name()) for process in processes)
        return processes

    def register_process_tree(self, root_process):
        """Add all current descendants of a managed process to supervision."""
        descendants = root_process.children(recursive=True)
        self.processes.extend(descendants)
        self.process_names.update((process, process.name()) for process in descendants)

    def terminate_all_processes(self):
        from lightllm.utils.envs_utils import get_env_start_args

        def kill_recursive(proc):
            try:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                for child in children:
                    logger.info(f"Killing child process {child.pid}")
                    child.kill()
                logger.info(f"Killing parent process {proc.pid}")
                parent.kill()
            except psutil.NoSuchProcess:
                logger.warning(f"Process {proc.pid} does not exist.")

        for proc in self.processes:
            if proc.is_running():
                kill_recursive(proc)
                proc.wait()

        # recover the gpu compute mode
        is_enable_mps = get_env_start_args().enable_mps
        if is_enable_mps:
            from lightllm.utils.device_utils import stop_mps

            stop_mps()
        logger.info("All processes terminated gracefully.")

    def setup_signal_handlers(self, http_server_process=None):
        def signal_handler(sig, _frame):
            if sig == signal.SIGINT:
                logger.info("Received SIGINT (Ctrl+C), forcing immediate exit...")
                if http_server_process is not None:
                    kill_recursive(http_server_process)

                self.terminate_all_processes()
                logger.info("All processes have been forcefully terminated.")
                sys.exit(0)

            if sig == signal.SIGTERM:
                logger.info("Received SIGTERM, shutting down gracefully...")
            else:
                logger.info("Received SIGHUP (terminal closed), shutting down gracefully...")

            if http_server_process is not None and http_server_process.poll() is None:
                http_server_process.send_signal(signal.SIGTERM)
                try:
                    http_server_process.wait(timeout=60)
                    logger.info("HTTP server exited gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("HTTP server did not exit in time, killing it...")
                    kill_recursive(http_server_process)

            self.terminate_all_processes()
            logger.info("All processes have been terminated gracefully.")
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)

        logger.info(f"start process pid {os.getpid()}")
        if http_server_process is not None:
            logger.info(f"http server pid {http_server_process.pid}")

    def supervise_processes(self, http_server_process=None):
        """Watch the HTTP server, when present, and all registered submodules.

        Signal-driven shutdown is handled by the launcher. Reaching an exited
        process here therefore means that the service can no longer operate
        correctly. Clean up the remaining process tree and raise so the container's
        main process exits with a non-zero status.
        """
        supervisor_interval_seconds = 5.0
        while True:
            if http_server_process is not None:
                http_return_code = http_server_process.poll()
                if http_return_code is not None:
                    message = f"HTTP server exited unexpectedly with return code {http_return_code}"
                    logger.error(message)
                    self._cleanup_after_process_failure(http_server_process)
                    raise RuntimeError(message)

            dead_processes = [
                process for process in self.processes if not process.is_running() or not is_process_active(process.pid)
            ]
            if dead_processes:
                dead_process_descriptions = []
                for process in dead_processes:
                    try:
                        exitcode = process.wait(timeout=0)
                    except psutil.TimeoutExpired:
                        exitcode = None
                    dead_process_descriptions.append(
                        f"name={self.process_names[process]} pid={process.pid} exitcode={exitcode}"
                    )
                dead_process_descriptions = ", ".join(dead_process_descriptions)
                message = f"Critical LightLLM submodule exited unexpectedly: {dead_process_descriptions}"
                logger.error(message)
                self._cleanup_after_process_failure(http_server_process)
                raise RuntimeError(message)

            time.sleep(supervisor_interval_seconds)

    def _cleanup_after_process_failure(self, http_server_process):
        """Best-effort cleanup before the launcher exits with a failure."""
        if http_server_process is not None and http_server_process.poll() is None:
            try:
                kill_recursive(http_server_process)
            except Exception:
                logger.exception("Failed to terminate the HTTP server process tree")

        try:
            self.terminate_all_processes()
        except Exception:
            logger.exception("Failed to terminate all LightLLM submodule processes")


def start_submodule_processes(start_funcs=[], start_args=[]):
    assert len(start_funcs) == len(start_args)
    pipe_readers = []
    processes = []
    for start_func, start_arg in zip(start_funcs, start_args):
        pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        process = mp.Process(
            target=start_func,
            args=start_arg + (pipe_writer,),
        )
        process.start()
        pipe_readers.append(pipe_reader)
        processes.append(process)

    # wait to ready
    for index, pipe_reader in enumerate(pipe_readers):
        init_state = pipe_reader.recv()
        if init_state != "init ok":
            logger.error(f"init func {start_funcs[index].__name__} : {str(init_state)}")
            for proc in processes:
                proc.kill()
            sys.exit(1)
        else:
            logger.info(f"init func {start_funcs[index].__name__} : {str(init_state)}")

    assert all([proc.is_alive() for proc in processes])
    return


def kill_recursive(proc):
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            logger.info(f"Killing child process {child.pid}")
            child.kill()
        logger.info(f"Killing parent process {proc.pid}")
        parent.kill()
    except psutil.NoSuchProcess:
        logger.warning(f"Process {proc.pid} does not exist.")


process_manager = SubmoduleManager()

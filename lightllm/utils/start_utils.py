import multiprocessing as mp
import os
import signal
import subprocess
import time

import psutil

from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import is_process_active

logger = init_logger(__name__)


# Waiting for an unrelated/re-parented zombie can otherwise block forever.
PROCESS_SHUTDOWN_WAIT_TIMEOUT_SECONDS = 5
HTTP_SERVER_SHUTDOWN_WAIT_TIMEOUT_SECONDS = 60


def _get_process_tree(root_process):
    """Return active descendants followed by the active root, keyed by PID."""
    try:
        root = psutil.Process(root_process.pid)
        descendants = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return []

    processes_by_pid = {}
    # Reverse the recursive list so that descendants are killed before parents.
    for process in list(reversed(descendants)) + [root]:
        try:
            if is_process_active(process.pid):
                processes_by_pid.setdefault(process.pid, process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            continue
    return list(processes_by_pid.values())


def kill_recursive(proc):
    """Best-effort, child-first SIGKILL of a local process tree.

    This intentionally does not wait: waiting for a zombie not owned by this
    process is the shutdown path that previously made Ctrl-C hang forever.
    """
    for process in _get_process_tree(proc):
        try:
            logger.info(f"Killing process {process.pid}")
            process.kill()
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            continue


class SubmoduleManager:
    def __init__(self):
        self.processes = []
        self.process_names = {}
        self.http_server_process = None
        self._signal_handlers_installed = False
        self._handling_signal = False

    def _register_process(self, process, resolve_pid=False):
        """Register one process once, by PID, as soon as it has started."""
        if process.pid is None:
            return None
        try:
            managed_process = psutil.Process(process.pid) if resolve_pid else process
            process_name = managed_process.name()
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            return None

        if any(existing_process.pid == managed_process.pid for existing_process in self.processes):
            return managed_process
        self.processes.append(managed_process)
        self.process_names[managed_process] = process_name
        return managed_process

    def start_submodule_processes(self, start_funcs=[], start_args=[]):
        assert len(start_funcs) == len(start_args)
        pipe_readers = []
        processes = []
        managed_processes = []

        try:
            for start_func, start_arg in zip(start_funcs, start_args):
                pipe_reader, pipe_writer = mp.Pipe(duplex=False)
                process = mp.Process(
                    target=start_func,
                    args=start_arg + (pipe_writer,),
                )
                process.start()
                # Register before waiting for initialization so Ctrl-C can clean
                # processes which are still starting up.
                managed_process = self._register_process(process, resolve_pid=True)
                if managed_process is not None:
                    managed_processes.append(managed_process)
                if hasattr(pipe_writer, "close"):
                    pipe_writer.close()
                pipe_readers.append(pipe_reader)
                processes.append(process)

            # Wait for all processes to initialize.
            for index, pipe_reader in enumerate(pipe_readers):
                init_state = pipe_reader.recv()
                if init_state != "init ok":
                    logger.error(f"init func {start_funcs[index].__name__} : {str(init_state)}")
                    raise RuntimeError(f"submodule {start_funcs[index].__name__} failed to initialize")
                logger.info(f"init func {start_funcs[index].__name__} : {str(init_state)}")

            if not all(process.is_alive() for process in processes):
                raise RuntimeError("submodule exited before initialization completed")
            return managed_processes
        except BaseException:
            # recv() may be interrupted by Ctrl-C or raise EOFError when a child
            # dies. All successfully-started children have already been managed.
            try:
                self.terminate_all_processes()
            except Exception:
                logger.exception("Failed to clean up submodules after initialization failure")
            raise
        finally:
            for pipe_reader in pipe_readers:
                try:
                    if hasattr(pipe_reader, "close"):
                        pipe_reader.close()
                except (OSError, EOFError):
                    pass

    def register_process_tree(self, root_process):
        """Add persistent LightLLM descendants to supervision.

        A managed process may create short-lived helper processes while loading
        models or compiling kernels. Those helpers retain a generic process name,
        while persistent LightLLM services set a ``lightllm::`` process title.
        """
        try:
            descendants = root_process.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            return

        for process in descendants:
            try:
                process_name = process.name()
                if not process_name.startswith("lightllm::"):
                    continue
                if not is_process_active(process.pid):
                    continue
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                continue
            self._register_process(process)

    def terminate_all_processes(self):
        """Kill all managed local process trees without indefinitely waiting."""
        from lightllm.utils.envs_utils import get_env_start_args

        processes_by_pid = {}
        for process in self.processes:
            for tree_process in _get_process_tree(process):
                processes_by_pid.setdefault(tree_process.pid, tree_process)

        processes_to_wait_for = list(processes_by_pid.values())
        for process in processes_to_wait_for:
            try:
                logger.info(f"Killing process {process.pid}")
                process.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                continue

        if processes_to_wait_for:
            try:
                _gone, alive = psutil.wait_procs(processes_to_wait_for, timeout=PROCESS_SHUTDOWN_WAIT_TIMEOUT_SECONDS)
                if alive:
                    logger.warning(
                        "Timed out waiting for processes to exit: %s",
                        ", ".join(str(process.pid) for process in alive),
                    )
            except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
                # A process may disappear between kill() and wait_procs().
                pass
            except Exception:
                logger.exception("Failed while waiting for submodule processes to exit")

        # Recover GPU compute mode, but failure here must not prevent launcher exit.
        try:
            is_enable_mps = get_env_start_args().enable_mps
            if is_enable_mps:
                from lightllm.utils.device_utils import stop_mps

                stop_mps()
        except Exception:
            logger.exception("Failed to restore GPU compute mode during shutdown")
        logger.info("All processes terminated gracefully.")

    def _terminate_http_server(self, graceful):
        http_server_process = self.http_server_process
        if http_server_process is None or http_server_process.poll() is not None:
            return
        try:
            if graceful:
                http_server_process.send_signal(signal.SIGTERM)
                try:
                    http_server_process.wait(timeout=HTTP_SERVER_SHUTDOWN_WAIT_TIMEOUT_SECONDS)
                    logger.info("HTTP server exited gracefully")
                    return
                except subprocess.TimeoutExpired:
                    logger.warning("HTTP server did not exit in time, killing it...")
            kill_recursive(http_server_process)
        except Exception:
            logger.exception("Failed to terminate the HTTP server process tree")

    def setup_signal_handlers(self, http_server_process=None):
        # The installed closure deliberately reads this field at signal time:
        # handlers are installed before submodules are launched, while Hypercorn
        # is only available later in the startup sequence.
        if http_server_process is not None:
            self.http_server_process = http_server_process

        if self._signal_handlers_installed:
            return

        def signal_handler(sig, _frame):
            if self._handling_signal:
                logger.warning("Received a second shutdown signal; exiting immediately")
                os._exit(1)
            self._handling_signal = True
            try:
                if sig == signal.SIGINT:
                    logger.info("Received SIGINT (Ctrl+C), forcing immediate exit...")
                    self._terminate_http_server(graceful=False)
                elif sig == signal.SIGTERM:
                    logger.info("Received SIGTERM, shutting down gracefully...")
                    self._terminate_http_server(graceful=True)
                else:
                    logger.info("Received SIGHUP (terminal closed), shutting down gracefully...")
                    self._terminate_http_server(graceful=True)

                self.terminate_all_processes()
                logger.info("All processes have been terminated.")
            except Exception:
                logger.exception("Shutdown cleanup failed")
            finally:
                # Do not let multiprocessing's atexit handler re-join a stuck
                # child after Ctrl-C. Cleanup above is intentionally best effort.
                os._exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)
        self._signal_handlers_installed = True

        logger.info(f"start process pid {os.getpid()}")
        if self.http_server_process is not None:
            logger.info(f"http server pid {self.http_server_process.pid}")

    def supervise_processes(self, http_server_process=None):
        """Watch the HTTP server, when present, and all registered submodules."""
        if http_server_process is not None:
            self.http_server_process = http_server_process
        supervisor_interval_seconds = 5.0
        while True:
            if self.http_server_process is not None:
                http_return_code = self.http_server_process.poll()
                if http_return_code is not None:
                    message = f"HTTP server exited unexpectedly with return code {http_return_code}"
                    logger.error(message)
                    self._cleanup_after_process_failure()
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
                self._cleanup_after_process_failure()
                raise RuntimeError(message)

            time.sleep(supervisor_interval_seconds)

    def _cleanup_after_process_failure(self):
        """Best-effort cleanup before the launcher exits with a failure."""
        self._terminate_http_server(graceful=False)
        try:
            self.terminate_all_processes()
        except Exception:
            logger.exception("Failed to terminate all LightLLM submodule processes")


def start_submodule_processes(start_funcs=[], start_args=[]):
    """Backward-compatible helper for callers that do not need supervision."""
    return SubmoduleManager().start_submodule_processes(start_funcs, start_args)


process_manager = SubmoduleManager()

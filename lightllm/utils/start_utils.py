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


# Waiting for an unrelated/re-parented zombie can otherwise block forever.
PROCESS_SHUTDOWN_WAIT_TIMEOUT_SECONDS = 5


class SubmoduleManager:
    def __init__(self):
        self.processes = []
        self.process_names = {}
        self.http_server_process = None
        self._handling_signal = False

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
                pipe_readers.append(pipe_reader)
                processes.append(process)
                try:
                    process.start()
                    # Register before waiting for initialization so Ctrl-C can clean
                    # processes which are still starting up.
                    managed_process = psutil.Process(process.pid)
                    self.processes.append(managed_process)
                    self.process_names[managed_process] = managed_process.name()
                    managed_processes.append(managed_process)
                finally:
                    pipe_writer.close()

            # Wait for all processes to initialize.
            for index, pipe_reader in enumerate(pipe_readers):
                init_state = pipe_reader.recv()
                if init_state != "init ok":
                    logger.error(f"init func {start_funcs[index].__name__} : {str(init_state)}")
                    raise SystemExit(1)
                logger.info(f"init func {start_funcs[index].__name__} : {str(init_state)}")

            assert all(process.is_alive() for process in processes)
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
                    pipe_reader.close()
                except (OSError, EOFError):
                    pass

    def register_process_tree(self, root_process):
        """Add persistent LightLLM descendants to supervision.

        A managed process may create short-lived helper processes while loading
        models or compiling kernels. Those helpers retain a generic process name,
        while persistent LightLLM services set a ``lightllm::`` process title.
        """
        for process in root_process.children(recursive=True):
            try:
                process_name = process.name()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                # A short-lived child may exit while the process tree is scanned.
                continue

            if not process_name.startswith("lightllm::"):
                continue

            if not process.is_running() or not is_process_active(process.pid):
                continue

            self.processes.append(process)
            self.process_names[process] = process_name

    def terminate_all_processes(self, wait_timeout=PROCESS_SHUTDOWN_WAIT_TIMEOUT_SECONDS):
        """Kill all managed local process trees without indefinitely waiting."""
        processes_by_pid = {}
        for process in self.processes:
            for tree_process in _get_process_tree(process):
                processes_by_pid.setdefault(tree_process.pid, tree_process)

        processes_to_wait_for = list(processes_by_pid.values())
        _kill_processes(processes_to_wait_for)

        if processes_to_wait_for and wait_timeout > 0:
            try:
                _gone, alive = psutil.wait_procs(processes_to_wait_for, timeout=wait_timeout)
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
            from lightllm.utils.envs_utils import get_env_start_args

            is_enable_mps = get_env_start_args().enable_mps
            if is_enable_mps:
                from lightllm.utils.device_utils import stop_mps

                stop_mps()
        except Exception:
            logger.exception("Failed to restore GPU compute mode during shutdown")
        logger.info("All processes terminated gracefully.")

    def setup_signal_handlers(self, http_server_process=None):
        from lightllm.utils.auto_shm_cleanup import get_auto_cleanup

        # Initialize shared-memory signal handling before the launcher takes over.
        shm_cleanup = get_auto_cleanup()
        # The installed closure deliberately reads this field at signal time:
        # handlers are installed before submodules are launched, while Hypercorn
        # is only available later in the startup sequence.
        if http_server_process is not None:
            self.http_server_process = http_server_process

        def signal_handler(sig, _frame):
            repeated_signal = self._handling_signal
            self._handling_signal = True
            try:
                if repeated_signal:
                    logger.warning("Received a second shutdown signal; forcing immediate cleanup")
                elif sig == signal.SIGINT:
                    logger.info("Received SIGINT (Ctrl+C), forcing immediate exit...")
                elif sig == signal.SIGTERM:
                    logger.info("Received SIGTERM, shutting down gracefully...")
                else:
                    logger.info("Received SIGHUP (terminal closed), shutting down gracefully...")

                if (
                    not repeated_signal
                    and sig != signal.SIGINT
                    and self.http_server_process is not None
                    and self.http_server_process.poll() is None
                ):
                    self.http_server_process.send_signal(signal.SIGTERM)
                    self.http_server_process.wait(timeout=60)
                    logger.info("HTTP server exited gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("HTTP server did not exit in time, killing it...")
            except Exception:
                logger.exception("Shutdown cleanup failed")
            finally:
                try:
                    for shutdown_signal in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                        signal.signal(shutdown_signal, signal.SIG_IGN)
                    self._cleanup_processes(
                        self.http_server_process,
                        wait_timeout=0 if repeated_signal else PROCESS_SHUTDOWN_WAIT_TIMEOUT_SECONDS,
                    )
                    logger.info("All processes have been terminated.")
                except Exception:
                    logger.exception("Failed to clean up processes during shutdown")
                finally:
                    try:
                        shm_cleanup.cleanup()
                    except Exception:
                        logger.exception("Failed to clean up shared memory during shutdown")
                    finally:
                        # Do not let multiprocessing's atexit handler re-join a stuck
                        # child after Ctrl-C. Cleanup above is intentionally best effort.
                        os._exit(1 if repeated_signal else 0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)

        logger.info(f"start process pid {os.getpid()}")
        if self.http_server_process is not None:
            logger.info(f"http server pid {self.http_server_process.pid}")

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
                    self._cleanup_processes(http_server_process)
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
                self._cleanup_processes(http_server_process)
                raise RuntimeError(message)

            time.sleep(supervisor_interval_seconds)

    def _cleanup_processes(self, http_server_process, wait_timeout=PROCESS_SHUTDOWN_WAIT_TIMEOUT_SECONDS):
        """Best-effort cleanup before the launcher exits."""
        try:
            if http_server_process is not None and http_server_process.poll() is None:
                kill_recursive(http_server_process)
        except Exception:
            logger.exception("Failed to terminate the HTTP server process tree")

        try:
            self.terminate_all_processes(wait_timeout=wait_timeout)
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


def _get_process_tree(root_process):
    try:
        root = psutil.Process(root_process.pid)
        descendants = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
        return []

    processes = []
    for process in list(reversed(descendants)) + [root]:
        try:
            if is_process_active(process.pid):
                processes.append(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            continue
    return processes


def _kill_processes(processes):
    for process in processes:
        try:
            logger.info(f"Killing process {process.pid}")
            process.kill()
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            continue


def kill_recursive(proc):
    _kill_processes(_get_process_tree(proc))


process_manager = SubmoduleManager()

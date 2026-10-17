import os
import ctypes
import signal
import subprocess
import sys
import time
import threading
import multiprocessing as mp
import psutil
from lightllm.utils.log_utils import init_logger
from lightllm.utils.process_check import is_process_active
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.service_shm_cleanup import start_launcher_shm_cleanup_process

logger = init_logger(__name__)


class ServiceReadiness:
    """Log transitions only; steady probes never produce log traffic."""

    def __init__(self, flag, args, address, started):
        self.flag, self.args, self.address, self.started = flag, args, address, started
        self.ready = False
        self.ever_ready = False
        self.waiting_logged = False

    def observe(self, status):
        if status == 200:
            if not self.ready:
                event = "service.recovered" if self.ever_ready else "service.ready"
                elapsed = round(time.monotonic() - self.started, 2)
                duration_key = "uptime_s" if self.ever_ready else "startup_s"
                fields = {
                    "event": event,
                    "run_mode": self.args.run_mode,
                    "model": self.args.model_name,
                    "address": self.address,
                    duration_key: elapsed,
                }
                logger.info(
                    "event=%s role=%s model=%s address=%s %s=%.2f",
                    event,
                    self.args.run_mode,
                    self.args.model_name,
                    self.address,
                    duration_key,
                    elapsed,
                    extra=fields,
                )
                self.flag.set_value(1)
                self.ready = self.ever_ready = True
        else:
            if self.ready:
                self.flag.set_value(0)
                self.ready = False
                logger.warning(
                    "event=service.unavailable reason=readiness_probe_failed status=%s",
                    status,
                    extra={"event": "service.unavailable", "reason": "readiness_probe_failed", "http_status": status},
                )
            elif status == 503 and not self.waiting_logged and not self.ever_ready:
                logger.info(
                    "event=service.waiting reason=pd_nodes_not_ready",
                    extra={"event": "service.waiting", "reason": "pd_nodes_not_ready"},
                )
                self.waiting_logged = True


class SubmoduleManager:
    def __init__(self):
        self.processes = []
        self.process_names = {}
        self.readiness_stop = threading.Event()

    def start_readiness_monitor(self, http_server_process, args):
        """Announce readiness after the bound HTTP endpoint and PD topology agree."""
        import requests
        from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
        from lightllm.utils.shm_port_args import get_shm_port_args

        ready_flag = SharedInt("service_ready")
        ready_flag.set_value(0)
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(args.host, args.host)
        host = f"[{host}]" if ":" in host else host
        address = f"http://{host}:{get_shm_port_args().port}"
        started = float(os.getenv("LIGHTLLM_SERVICE_START_TIME", str(time.monotonic())))
        self.readiness_stop.clear()

        def monitor():
            state = ServiceReadiness(ready_flag, args, address, started)
            with requests.Session() as session:
                session.trust_env = False
                while not self.readiness_stop.is_set() and http_server_process.poll() is None:
                    try:
                        response = session.get(address + "/readiness", timeout=1)
                        state.observe(response.status_code)
                    except requests.RequestException:
                        state.observe(None)
                    self.readiness_stop.wait(0.5)

        threading.Thread(target=monitor, name="service-readiness", daemon=True).start()

    def start_submodule_processes(self, start_funcs=[], start_args=[]):
        assert len(start_funcs) == len(start_args)
        pipe_readers = []
        processes = []
        managed_processes = []

        for start_func, start_arg in zip(start_funcs, start_args):
            pipe_reader, pipe_writer = mp.Pipe(duplex=False)
            process = mp.Process(
                target=start_func,
                args=start_arg + (pipe_writer,),
            )
            process.start()
            pipe_readers.append(pipe_reader)
            processes.append(process)
            # 初始化完成前也可能收到退出信号，因此子进程启动后立即纳入管理。
            managed_process = psutil.Process(process.pid)
            managed_processes.append(managed_process)
            self.processes.append(managed_process)
            self.process_names[managed_process] = managed_process.name()

        # Wait for all processes to initialize
        for index, pipe_reader in enumerate(pipe_readers):
            init_state = pipe_reader.recv()
            if init_state != "init ok":
                logger.error(f"init func {start_funcs[index].__name__} : {str(init_state)}")
                for proc in processes:
                    proc.kill()
                sys.exit(1)
            else:
                logger.debug("Worker initialized: component=%s", start_funcs[index].__name__)

        assert all([proc.is_alive() for proc in processes])
        return managed_processes

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

    def terminate_all_processes(self):
        from lightllm.utils.envs_utils import get_env_start_args

        self.readiness_stop.set()
        for proc in self.processes:
            if proc.is_running():
                kill_recursive(proc)

        # Signal every process before waiting. Registered router descendants may
        # remain zombies under another parent, so bound the total wait time.
        _, alive = psutil.wait_procs(self.processes, timeout=5)
        alive_pids = [proc.pid for proc in alive if is_process_active(proc.pid)]
        if alive_pids:
            logger.warning(f"Processes still alive after SIGKILL: {alive_pids}")

        # recover the gpu compute mode
        is_enable_mps = get_env_start_args().enable_mps
        if is_enable_mps:
            from lightllm.utils.device_utils import stop_mps

            stop_mps()
        if not alive_pids:
            logger.info("event=service.stopped cleanup=complete")

    def setup_exit_controller(self):
        """启动 launcher 的独立资源清理进程。

        在 service name 和启动参数写入环境后、创建共享内存或启动子进程前调用。
        launcher 退出后由独立进程回收资源。
        """
        if sys.platform == "linux":
            # 接管 router 退出后的 model/KV 后代，使现有 wait_procs 能真正回收它们，
            # 避免交给不执行 wait 的容器 PID 1（如 sleep infinity）。
            prctl = ctypes.CDLL(None, use_errno=True).prctl
            prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
            if prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
                error = ctypes.get_errno()
                raise OSError(error, f"Failed to enable child subreaper: {os.strerror(error)}")
        start_launcher_shm_cleanup_process(get_unique_server_name())

    def setup_signal_handlers(self, http_server_process=None):
        """在子进程启动完成后安装退出信号处理函数，覆盖启动阶段的处理函数。"""

        def signal_handler(sig, _frame):
            self.readiness_stop.set()
            if sig == signal.SIGINT:
                logger.info("Received SIGINT (Ctrl+C), forcing immediate exit...")
                if http_server_process is not None:
                    kill_recursive(http_server_process)

                self.terminate_all_processes()
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
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)

        logger.debug("Launcher pid=%s", os.getpid())
        if http_server_process is not None:
            logger.debug("HTTP server pid=%s", http_server_process.pid)

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
        self.readiness_stop.set()
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
            logger.debug("Worker initialized: component=%s", start_funcs[index].__name__)

    assert all([proc.is_alive() for proc in processes])
    return


def kill_recursive(proc):
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        logger.debug(f"Process {proc.pid} does not exist.")
        return

    for process in [*reversed(children), parent]:
        try:
            logger.debug(f"Killing process {process.pid}")
            process.kill()
        except psutil.NoSuchProcess:
            # A concurrent exit must not skip the remaining children or parent.
            logger.debug(f"Process {process.pid} does not exist.")


process_manager = SubmoduleManager()

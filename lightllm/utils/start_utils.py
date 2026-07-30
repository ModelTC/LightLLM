import sys
import multiprocessing as mp
import psutil
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class SubmoduleManager:
    _TERMINATE_TIMEOUT = 5

    def __init__(self):
        self.processes = []

    def start_submodule_processes(self, start_funcs=[], start_args=[]):
        assert len(start_funcs) == len(start_args)
        pipe_readers = []
        processes = []

        try:
            for start_func, start_arg in zip(start_funcs, start_args):
                pipe_reader, pipe_writer = mp.Pipe(duplex=False)
                process = mp.Process(
                    target=start_func,
                    args=start_arg + (pipe_writer,),
                )
                process.start()
                pipe_readers.append(pipe_reader)
                processes.append(process)
                self.processes.append(process)

            for index, pipe_reader in enumerate(pipe_readers):
                init_state = pipe_reader.recv()
                if init_state != "init ok":
                    raise RuntimeError(f"init func {start_funcs[index].__name__} : {str(init_state)}")
                logger.info(f"init func {start_funcs[index].__name__} : {str(init_state)}")

            assert all([proc.is_alive() for proc in processes])
        except BaseException:
            self.terminate_all_processes(graceful=False)
            raise
        return

    def _terminate_processes(self, processes, graceful):
        process_by_pid = {}
        for proc in processes:
            if proc.pid is None:
                continue
            try:
                parent = psutil.Process(proc.pid)
                process_by_pid[parent.pid] = parent
                for child in parent.children(recursive=True):
                    process_by_pid[child.pid] = child
            except psutil.NoSuchProcess:
                continue

        process_tree = list(process_by_pid.values())
        for process in reversed(process_tree):
            try:
                if graceful:
                    process.terminate()
                else:
                    process.kill()
            except psutil.NoSuchProcess:
                pass

        if graceful:
            _, alive = psutil.wait_procs(process_tree, timeout=self._TERMINATE_TIMEOUT)
            for process in alive:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(alive, timeout=self._TERMINATE_TIMEOUT)

        for proc in processes:
            if proc.pid is not None:
                proc.join(timeout=1)

    def terminate_all_processes(self, graceful=True):
        from lightllm.utils.envs_utils import get_env_start_args

        self._terminate_processes(self.processes, graceful)
        self.processes.clear()

        # recover the gpu compute mode
        try:
            is_enable_mps = get_env_start_args().enable_mps
        except (AttributeError, KeyError):
            is_enable_mps = False
        if is_enable_mps:
            from lightllm.utils.device_utils import stop_mps

            stop_mps()
        logger.info("All processes terminated.")


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

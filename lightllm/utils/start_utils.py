import sys
import multiprocessing as mp
import psutil
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class SubmoduleManager:
    def __init__(self):
        self.processes = []

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
        self.processes.extend(processes)
        return

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
            if proc.is_alive():
                kill_recursive(proc)
                proc.join()

        # recover the gpu compute mode
        is_enable_mps = get_env_start_args().enable_mps
        if is_enable_mps:
            from lightllm.utils.device_utils import stop_mps

            stop_mps()
        logger.info("All processes terminated gracefully.")


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

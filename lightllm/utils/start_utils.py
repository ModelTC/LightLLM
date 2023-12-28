import sys
import multiprocessing as mp
from lightllm.utils.log_utils import init_logger
logger = init_logger(__name__)

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
        if init_state != 'init ok':
            logger.error(f"init func {start_funcs[index].__name__} : {str(init_state)}")
            for proc in processes:
                proc.kill()
            sys.exit(1)
        else:
            logger.info(f"init func {start_funcs[index].__name__} : {str(init_state)}")
    
    assert all([proc.is_alive() for proc in processes])
    return

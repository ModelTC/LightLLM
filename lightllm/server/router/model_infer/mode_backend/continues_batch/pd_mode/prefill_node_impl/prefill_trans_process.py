import torch
import time
import sys
import torch.multiprocessing as mp
from typing import List, Dict
from lightllm.utils.log_utils import init_logger
from lightllm.common.mem_manager import MemoryManager
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.device_utils import kv_trans_use_p2p, init_p2p

logger = init_logger(__name__)


# device_index 是用来指示，当前传输进程使用的用于数据传输的显卡id
# 当模型是多卡推理的时候，需要传输的 kv 需要先移动到 device_index
# 指定的显卡上，然后再进行传输，因为torch nccl 限制了只能操作一张显卡上的数据


def _init_env(
    args,
    device_index: int,
    nccl_ip,
    nccl_port,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
):
    import os

    # os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_MAX_NCHANNELS"] = "2"
    os.environ["NCCL_NSOCKS_PER_CHANNEL"] = "1"
    os.environ["NCCL_SOCKET_NTHREADS"] = "1"
    torch.backends.cudnn.enabled = False

    try:
        # 注册graceful 退出的处理
        from lightllm.utils.graceful_utils import graceful_registry
        import inspect

        graceful_registry(inspect.currentframe().f_code.co_name)
        torch.cuda.set_device(device_index)

        task_out_queue.put("proc_start")
        mem_managers: List[MemoryManager] = [mem_queue.get(timeout=60) for mem_queue in mem_queues]
        assert len(mem_managers) == args.tp
        task_out_queue.put("get_mem_managers_ok")
        import torch.distributed as dist
        from datetime import timedelta

        dist.init_process_group(
            "nccl", init_method=f"tcp://{nccl_ip}:{nccl_port}", rank=0, world_size=2, timeout=timedelta(seconds=60)
        )
        task_out_queue.put("nccl_ok")
        while True:
            move_tasks: List[KVMoveTask] = task_in_queue.get()
            total_move_kv_len = sum([task.move_kv_len for task in move_tasks])
            try:
                start = time.time()
                if total_move_kv_len != 0:
                    logger.info(f"trans start: {move_tasks[0].to_prefill_log_info()}")
                    cur_mem = mem_managers[device_index]
                    if kv_trans_use_p2p():
                        cur_mem.send_to_decode_node_p2p(move_tasks, mem_managers, args.dp)
                    else:
                        cur_mem.send_to_decode_node(move_tasks, mem_managers, args.dp)
                    logger.info(f"trans finished: {move_tasks[0].to_prefill_log_info()} move len: {total_move_kv_len}")
                torch.cuda.synchronize()
                logger.info(f"trans cost time: {(time.time() - start)}, {move_tasks[0].to_prefill_log_info()}")
                task_out_queue.put("ok")
            except BaseException as e:
                logger.exception(str(e))
                task_out_queue.put("fail")
                raise e
    except BaseException as e:
        logger.exception(str(e))
        sys.exit(-1)
    return


def start_prefill_trans_process(
    args,
    device_index: int,
    nccl_ip,
    nccl_port,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
):
    proc = mp.Process(
        target=_init_env, args=(args, device_index, nccl_ip, nccl_port, task_in_queue, task_out_queue, mem_queues)
    )
    proc.start()
    assert proc.is_alive()
    logger.info(f"trans kv process start, nccl_ip: {nccl_ip}, nccl_port: {nccl_port}")
    return proc

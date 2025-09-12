import inspect
import pickle
import torch.multiprocessing as mp
from typing import List, Dict, Optional, Tuple, Union, Callable
from lightllm.utils.log_utils import init_logger
from lightllm.server.pd_io_struct import NIXLChunckedTransTask, NixlUpKVStatus, NIXLDecodeNodeInfo
from lightllm.server.core.objs import StartArgs
from lightllm.utils.graceful_utils import graceful_registry
from ..trans_process_obj import KVTransProcess
from ..base_kv_move_manager import BaseKVMoveManager

logger = init_logger(__name__)

def start_decode_kv_move_manager_process(args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("decode kv move manager process started")
    return

def _init_env(args, info_queue: mp.Queue, mem_queues: List[mp.Queue], event: mp.Event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    
    from .decode_trans_process import start_decode_trans_process
    manager = DecodeKVMoveManager(args=args, 
                                   info_queue=info_queue,
                                   mem_queues=mem_queues,
                                   start_trans_process_func=start_decode_trans_process)
    event.set()
    # 阻塞等待子线程退出
    manager.dispatch_task.join()
    return


class DecodeKVMoveManager(BaseKVMoveManager):
    def __init__(self, args: StartArgs, info_queue: mp.Queue, mem_queues: List[mp.Queue], start_trans_process_func: Callable):
        super().__init__(args=args,
                         info_queue=info_queue,
                         mem_queues=mem_queues,
                         start_trans_process_func=start_trans_process_func)

        from .up_status import start_up_kv_status_process

        self.up_status_in_queue = mp.Queue()
        self.up_status_out_queue = mp.Queue()
        start_up_kv_status_process(self.args, self.up_status_in_queue, self.up_status_out_queue)
        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求, 转发给 KV_TRANS_PROCESS
    # ==================================================================================

    def task_dispatcher_loop(self):
        try:
            # 获取任务，并分发给相关卡的处理队列
            while True:
                task:NIXLChunckedTransTask = self.info_queue.get()

                device_id = task.dst_device_id
                try:
                    trans_process: KVTransProcess = self.kv_trans_processes[device_id]
                    trans_process.task_in_queue.put(task)
                    agent_meta_data = trans_process.agent_meta_data
                    if task.is_first_chuncked:
                        decode_node_info = NIXLDecodeNodeInfo(
                            decode_node_id=self.args.pd_node_id,
                            pd_master_node_id=task.pd_master_node_id,
                            agent_name=agent_meta_data.agent_name,
                            agent_metadata=agent_meta_data.agent_metadata,
                            num_pages=agent_meta_data.num_pages,
                            page_reg_desc=agent_meta_data.page_reg_desc,
                            ready_kv_len=task.start_kv_index,
                        )

                        up_kv_status = NixlUpKVStatus(
                            group_request_id=task.request_id,
                            pd_master_node_id=task.pd_master_node_id,
                            nixl_params=pickle.dumps(decode_node_info)
                        )

                        self.up_status_in_queue.put(up_kv_status)

                except BaseException as e:
                    logger.exception(str(e))

        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e
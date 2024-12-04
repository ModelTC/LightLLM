import asyncio
import rpyc
import sys
import os
import gc
import signal
import copy
import numpy as np
import psutil
from dataclasses import dataclass
from typing import List, Dict
from lightllm.utils.log_utils import init_logger
from .prefill_infer_rpyc import PDPrefillInferRpcServer
from lightllm.common.mem_manager import MemoryManager
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.net_utils import find_available_port
from lightllm.utils.retry_utils import retry
from rpyc.utils.classic import obtain
from rpyc import AsyncResult
from lightllm.utils.net_utils import get_hostname_ip

logger = init_logger(__name__)


class DecodeBusyError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


@dataclass
class TransProcessObj:
    decode_node_id: str = None
    rpyc_conn: object = None  # rpyc_con 的连接对象
    process: mp.Process = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    nccl_ip: str = None
    nccl_port: str = None
    device_index: str = None  # 使用的gpu序号

    def create(
        self, decode_node_id: str, decode_node_ip: str, decode_node_rpyc_port: int, manager: "PrefillKVMoveManager"
    ):
        con = rpyc.connect(
            host=decode_node_ip, port=decode_node_rpyc_port, config={"allow_pickle": True}, keepalive=True
        )
        nccl_ip = manager.host_ip
        nccl_port = find_available_port(manager.args.pd_p_allowed_port_min, manager.args.pd_p_allowed_port_max)
        if nccl_port is None:
            raise Exception("no pd nccl port can be used")

        from .prefill_trans_process import start_prefill_trans_process

        device_index = manager.get_next_device_index()  # 分配 trans 进程使用的显卡
        task_in_queue = mp.Queue()
        task_out_queue = mp.Queue()
        proc = start_prefill_trans_process(
            manager.args, device_index, nccl_ip, nccl_port, task_in_queue, task_out_queue, manager.mem_queues
        )
        assert task_out_queue.get(timeout=30) == "proc_start"
        for obj in manager.infer_rpyc_objs:
            obj.put_mem_manager_to_mem_queue()
        assert task_out_queue.get(timeout=60) == "get_mem_managers_ok"
        prefill_node_id = manager.args.pd_node_id
        con.root.build_trans_process(prefill_node_id, nccl_ip, nccl_port)  # 异步调用, 让decode节点建立与prefill节点进行nccl通信的进程
        assert task_out_queue.get(timeout=60) == "nccl_ok"

        self.decode_node_id = decode_node_id
        self.rpyc_conn = con
        self.process = proc
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.nccl_port = nccl_port
        self.nccl_ip = nccl_ip
        self.device_index = device_index
        return

    def check_trans_process(self):
        process = psutil.Process(self.process.pid)
        if not (process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
            raise Exception(f"trans process: {self.process.pid} is dead")
        return

    def __del__(self):
        # 强制关闭连接和杀掉传输进程
        if self.process is not None:
            logger.warning(f"prefill trans process {self.process.pid} is killed")
            os.kill(self.process.pid, signal.SIGKILL)
        pass


class PrefillKVMoveManager:
    def __init__(self, args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
        self.args = args
        self.dp_size = self.args.dp
        self.info_queue = info_queue
        self.mem_queues = mem_queues
        self.infer_rpyc_objs: List[PDPrefillInferRpcServer] = []
        self.node_id_to_trans_obj: Dict[str, TransProcessObj] = {}
        for port in self.args.pd_tp_infer_rpyc_ports:
            con = retry(max_attempts=20, wait_time=2)(rpyc.connect)("localhost", port, config={"allow_pickle": True})
            self.infer_rpyc_objs.append(con.root)
            logger.info(f"rpyc connect to infer rpyc port: {port} ok")
        self.host_ip = get_hostname_ip()
        if self.host_ip is None:
            self.host_ip = args.host
        return

    def get_next_device_index(self):
        counts = [0 for _ in range(self.args.tp)]
        for obj in self.node_id_to_trans_obj.values():
            counts[obj.device_index] += 1
        device_index = int(np.argmin(counts))
        return device_index

    def get_trans_obj(self, task: KVMoveTask):
        if task.decode_node.node_id not in self.node_id_to_trans_obj:
            # 先遍历删除老的不能用的连接
            self.remove_dead_trans_obj()
            trans_obj = TransProcessObj()
            trans_obj.create(task.decode_node.node_id, task.decode_node.ip, task.decode_node.rpyc_port, self)
            self.node_id_to_trans_obj[task.decode_node.node_id] = trans_obj
        return self.node_id_to_trans_obj[task.decode_node.node_id]

    def remove_dead_trans_obj(self):
        del_node_ids = []
        for node_id, t_obj in self.node_id_to_trans_obj.items():
            try:
                t_obj.rpyc_conn.root.check_alive()
            except BaseException as e:
                logger.error(f"check error {str(e)}")
                del_node_ids.append(node_id)
        for node_id in del_node_ids:
            self.node_id_to_trans_obj.pop(node_id)

        if len(del_node_ids) != 0:
            gc.collect()
        return

    def handle_loop(self):
        try:
            while True:
                move_task = self.info_queue.get()
                if not isinstance(move_task, KVMoveTask):
                    logger.error("receive type is not KVMoveTask")
                    sys.exit(-1)

                logger.info(f"prefill node get task {move_task.to_prefill_log_info()}")
                try:
                    trans_obj = self.get_trans_obj(move_task)
                    # 申请传输
                    trans_move_task = copy.copy(move_task)
                    # 不需要发送prefill节点的token index信息给decode节点
                    trans_move_task.prefill_token_indexes = None
                    # 申请发送，并收到发送长度 move_kv_len.
                    move_kv_len = obtain(trans_obj.rpyc_conn.root.request_data_transfer(trans_move_task))
                    # 代表对方已经很繁忙了，放弃这次发送，改为用
                    if move_kv_len is None:
                        raise DecodeBusyError(f"decode_node_id {trans_obj.decode_node_id} is busy")

                    move_task.move_kv_len = move_kv_len
                    logger.info(f"prefill node request_data_transfer ok, {move_task.to_prefill_log_info()}")
                    # 开始传输直到完成
                    trans_obj.task_in_queue.put(move_task, timeout=10)
                    assert trans_obj.task_out_queue.get(timeout=30) == "ok"
                    logger.info(f"prefill node transfer data ok, req_id: {move_task.id()}")

                except DecodeBusyError as e:
                    logger.error(str(e))

                except BaseException as e:
                    logger.exception(str(e))
                    logger.error(f"kv move task {move_task.to_prefill_log_info()} has error, remove the trans_obj")
                    self.node_id_to_trans_obj.pop(move_task.decode_node.node_id, None)

                finally:
                    # 去引用否则进程无法杀掉
                    trans_obj = None
                    # 解除对prefill token的占用状态。
                    self._remove_req_refs_from_prompt_cache(move_task)

        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e

    def _remove_req_refs_from_prompt_cache(self, move_task: KVMoveTask):
        futures: List[AsyncResult] = []
        if self.dp_size == 1:
            infer_rpycs = self.infer_rpyc_objs
        else:
            infer_rpycs = [self.infer_rpyc_objs[move_task.prefill_dp_index]]

        for infer_rpyc in infer_rpycs:
            futures.append(rpyc.async_(infer_rpyc.remove_req_refs_from_prompt_cache)(move_task.group_request_id))
        asyncio.run(self.wait_all_future_finish(futures))
        return

    async def wait_all_future_finish(self, futures: List[AsyncResult]):
        await asyncio.gather(*[asyncio.to_thread(future.wait) for future in futures])
        return


def _init_env(args, info_queue: mp.Queue, mem_queues: List[mp.Queue], event: mp.Event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    manager = PrefillKVMoveManager(args, info_queue, mem_queues)
    event.set()
    # 进入主循环
    manager.handle_loop()
    return


def start_prefill_kv_move_manager_process(args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move manager process started")
    return

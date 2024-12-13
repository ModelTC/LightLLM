import asyncio
import time
import rpyc
import sys
import os
import gc
import signal
import copy
import numpy as np
import psutil
import threading
import random
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

KV_MOVE_MAX_NUM = 16

logger = init_logger(__name__)


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
    move_task_queue: List[KVMoveTask] = None
    manager: "PrefillKVMoveManager" = None
    has_error: bool = False

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
        manager._put_mem_manager_to_mem_queue()
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
        self.manager = manager
        self.move_task_queue_lock = threading.Lock()
        self.move_task_queue = []

        # 启动处理任务的线程
        self.thread = threading.Thread(target=self.handle_loop, daemon=True)
        self.thread.start()
        return

    def check_trans_process(self):
        process = psutil.Process(self.process.pid)
        if not (process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
            self.has_error = True
            raise Exception(f"trans process: {self.process.pid} is dead")
        return

    def put(self, move_task):
        if self.has_error:
            raise Exception(f"trans obj {self.decode_node_id} has error, can not put")
        with self.move_task_queue_lock:
            self.move_task_queue.append(move_task)

    def get_tasks(self):
        with self.move_task_queue_lock:
            move_tasks: List[KVMoveTask] = self.move_task_queue[0:KV_MOVE_MAX_NUM]
            self.move_task_queue = self.move_task_queue[KV_MOVE_MAX_NUM:]
        return move_tasks

    def clear_tasks(self):
        with self.move_task_queue_lock:
            for move_task in self.move_task_queue:
                self.manager._remove_req_refs_from_prompt_cache(move_task)
            self.move_task_queue = []

    def handle_loop(self):
        while not self.has_error:
            if len(self.move_task_queue) == 0:
                time.sleep(0.01)
                continue

            move_tasks: List[KVMoveTask] = self.get_tasks()
            if len(move_tasks) == 0:
                continue

            handle_list: List[KVMoveTask] = []
            not_handle_list: List[KVMoveTask] = []

            try:
                # random to check stats
                if random.randint(0, 20) == 10:
                    self.check_trans_process()
                    self.rpyc_conn.root.check_alive()

                for move_task in move_tasks:
                    logger.info(
                        f"prefill node get task {move_task.to_prefill_log_info()} "
                        f"queue time {move_task.get_cost_time()} s "
                        f"queue leff size {len(self.move_task_queue)} "
                    )

                trans_move_tasks = [copy.copy(move_task) for move_task in move_tasks]
                for trans_move_task in trans_move_tasks:
                    trans_move_task.prefill_token_indexes = None

                mark_start = time.time()
                move_kv_lens = self.rpyc_conn.root.request_data_transfer(trans_move_tasks)
                request_data_transfer_cost_time = time.time() - mark_start

                logger.info(
                    f"prefill node request_data_transfer ok, {move_tasks[0].to_prefill_log_info()}"
                    f" cost time: {request_data_transfer_cost_time} s"
                )

                for i, move_task in enumerate(move_tasks):
                    if move_kv_lens[i] is not None:
                        move_task.move_kv_len = move_kv_lens[i]
                        handle_list.append(move_task)
                    else:
                        not_handle_list.append(move_task)

                with self.manager.device_locks[self.device_index]:

                    for move_task in handle_list:
                        self.task_in_queue.put(move_task, timeout=10)

                    for move_task in handle_list:
                        assert self.task_out_queue.get(timeout=60) == "ok"
                        self.manager._remove_req_refs_from_prompt_cache(move_task)
                        move_tasks.remove(move_task)
                        logger.info(
                            f"prefill node transfer data ok, req_id: {move_task.id()}"
                            f" cost total time: {move_task.get_cost_time()} s"
                        )

                for move_task in not_handle_list:
                    logger.info(f"prefill node kv move task req_id: {move_task.id()} not send, decode is busy")

            except BaseException as e:
                logger.exception(str(e))
                logger.error(f"tran obj id {self.decode_node_id} has error, remove the trans_obj")
                self.has_error = True
                self.manager.remove_trans_obj(self.decode_node_id)
                # 将队列中没处理的数据全部清空
                self.clear_tasks()

            finally:
                for move_task in move_tasks:
                    self.manager._remove_req_refs_from_prompt_cache(move_task)

        logger.error(f"trans thread, decode id {self.decode_node_id} device_index {self.device_index} thread quit")
        return

    def __del__(self):
        self.has_error = True
        self.clear_tasks()

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
            socket_path = f"/tmp/prefill_node_infer_rpyc_{port}"
            from rpyc.utils.factory import unix_connect

            con = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})
            self.infer_rpyc_objs.append(con.root)
            logger.info(f"rpyc connect to infer rpyc port: {port} ok")
        self.host_ip = get_hostname_ip()
        if self.host_ip is None:
            self.host_ip = args.host

        self.infer_rpyc_lock = threading.Lock()
        # 需要每个卡有一个锁来规划每次只能有一个tran obj 操作对应显卡上的传输任务。
        self.device_locks = [threading.Lock() for _ in range(self.args.tp)]
        return

    def get_next_device_index(self):
        counts = [0 for _ in range(self.args.tp)]
        for obj in self.node_id_to_trans_obj.values():
            counts[obj.device_index] += 1
        device_index = int(np.argmin(counts))
        return device_index

    def get_trans_obj(self, task: KVMoveTask):
        self.remove_dead_trans_obj()
        if task.decode_node.node_id not in self.node_id_to_trans_obj:
            trans_obj = TransProcessObj()
            trans_obj.create(task.decode_node.node_id, task.decode_node.ip, task.decode_node.rpyc_port, self)
            self.node_id_to_trans_obj[task.decode_node.node_id] = trans_obj
        return self.node_id_to_trans_obj[task.decode_node.node_id]

    def remove_trans_obj(self, decode_node_id):
        if decode_node_id in self.node_id_to_trans_obj:
            trans_obj = self.node_id_to_trans_obj.pop(decode_node_id, None)
            if trans_obj is not None:
                trans_obj.has_error = True
        return

    def remove_dead_trans_obj(self):
        del_node_ids = []
        for node_id, t_obj in self.node_id_to_trans_obj.items():
            if t_obj.has_error or (not t_obj.thread.is_alive()):
                t_obj.has_error = True
                del_node_ids.append(node_id)

        for node_id in del_node_ids:
            self.node_id_to_trans_obj.pop(node_id, None)

        if len(del_node_ids) != 0:
            gc.collect()
        return

    def task_dispatcher_loop(self):
        try:
            # 获取任务，并分发给相关卡的处理队列
            while True:
                move_task: KVMoveTask = self.info_queue.get()
                try:
                    trans_obj = self.get_trans_obj(move_task)
                    trans_obj.put(move_task)
                except BaseException as e:
                    logger.exception(str(e))
                    self._remove_req_refs_from_prompt_cache(move_task)
                finally:
                    trans_obj = None

        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e

    def _remove_req_refs_from_prompt_cache(self, move_task: KVMoveTask):
        futures: List[AsyncResult] = []
        if self.dp_size == 1:
            infer_rpycs = self.infer_rpyc_objs
        else:
            infer_rpycs = [self.infer_rpyc_objs[move_task.prefill_dp_index]]
        with self.infer_rpyc_lock:
            for infer_rpyc in infer_rpycs:
                futures.append(rpyc.async_(infer_rpyc.remove_req_refs_from_prompt_cache)(move_task.group_request_id))
            asyncio.run(self.wait_all_future_finish(futures))
        return

    def _put_mem_manager_to_mem_queue(self):
        with self.infer_rpyc_lock:
            for obj in self.infer_rpyc_objs:
                obj.put_mem_manager_to_mem_queue()
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
    manager.task_dispatcher_loop()
    return


def start_prefill_kv_move_manager_process(args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move manager process started")
    return

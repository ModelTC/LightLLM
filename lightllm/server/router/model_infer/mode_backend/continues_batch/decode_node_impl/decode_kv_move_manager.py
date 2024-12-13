import rpyc
import random
import asyncio
import sys
import os
import signal
import time
import psutil
import threading
from rpyc.utils.classic import obtain
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from rpyc import ThreadedServer
from lightllm.utils.log_utils import init_logger
from .decode_infer_rpyc import PDDecodeInferRpcServer
from lightllm.common.mem_manager import MemoryManager
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask, UpKVStatus
from lightllm.utils.retry_utils import retry
import numpy as np
import queue
from rpyc import AsyncResult

logger = init_logger(__name__)

thread_local_data = threading.local()

KV_MOVE_MAX_NUM = 16


@dataclass
class TransProcessObj:
    prefill_node_id: str = None
    process: mp.Process = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    nccl_ip: str = None
    nccl_port: str = None
    device_index: int = None
    move_task_queue: List[KVMoveTask] = None
    manager: "DecodeKVMoveManager" = None
    has_error: bool = False

    def create(self, prefill_node_id: str, nccl_ip: str, nccl_port: int, manager: "DecodeKVMoveManager"):
        from .decode_trans_process import start_decode_trans_process

        task_in_queue = mp.Queue()
        task_out_queue = mp.Queue()
        device_index = manager.get_next_device_index()
        proc = start_decode_trans_process(
            manager.args, device_index, nccl_ip, nccl_port, task_in_queue, task_out_queue, manager.mem_queues
        )
        assert task_out_queue.get(timeout=30) == "proc_start"
        manager._put_mem_manager_to_mem_queue()
        assert task_out_queue.get(timeout=60) == "get_mem_managers_ok"
        assert task_out_queue.get(timeout=60) == "nccl_ok"

        self.prefill_node_id = prefill_node_id
        self.process = proc
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.nccl_ip = nccl_ip
        self.nccl_port = nccl_port
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
            raise Exception(f"trans obj {self.prefill_node_id} has error, can not put")
        with self.move_task_queue_lock:
            self.move_task_queue.append(move_task)

    def put_list(self, move_tasks):
        if self.has_error:
            raise Exception(f"trans obj {self.prefill_node_id} has error, can not put")
        with self.move_task_queue_lock:
            self.move_task_queue.extend(move_tasks)

    def get_tasks(self):
        with self.move_task_queue_lock:
            move_tasks: List[KVMoveTask] = self.move_task_queue[0:KV_MOVE_MAX_NUM]
            self.move_task_queue = self.move_task_queue[KV_MOVE_MAX_NUM:]
        return move_tasks

    def clear_tasks(self):
        with self.move_task_queue_lock:
            for move_task in self.move_task_queue:
                self.manager._fail_to_realese_forzen_tokens(move_task)
                logger.error(f"decode kv move task {move_task.to_decode_log_info()} has error, unforzen tokens")
            self.move_task_queue = []

    def handle_loop(self):
        while not self.has_error:
            if len(self.move_task_queue) == 0:
                time.sleep(0.01)
                continue

            move_tasks: List[KVMoveTask] = self.get_tasks()
            if len(move_tasks) == 0:
                continue

            for task in move_tasks:
                logger.info(f"deocode node get task {task.to_decode_log_info()}")

            try:
                # random to check stats
                if random.randint(0, 20) == 10:
                    self.check_trans_process()
                iter_move_tasks = move_tasks.copy()
                with self.manager.device_locks[self.device_index]:
                    for task in iter_move_tasks:
                        self.task_in_queue.put(task, timeout=10)
                    for task in iter_move_tasks:
                        assert self.task_out_queue.get(timeout=60) == "ok"
                        logger.info(f"deocode node transfer kv ok {task.to_decode_log_info()}")
                        # 成功了将 token 放入 prompt cache 中
                        self.manager._put_kv_received_to_radix_cache(task)
                        move_tasks.remove(task)
                        logger.info(f"decode node put kv to radix cache ok, req_id: {task.id()}")
                        self.manager.up_status_in_queue.put(
                            UpKVStatus(group_request_id=task.group_request_id, dp_index=task.decode_dp_index)
                        )
                        logger.info("decode node up kv status finished")

            except BaseException as e:
                logger.exception(str(e))
                self.has_error = True
                self.clear_tasks()
                self.manager.remove_trans_obj(self.prefill_node_id)

            finally:
                for move_task in move_tasks:
                    self.manager._fail_to_realese_forzen_tokens(move_task)

        logger.error(f"trans thread, prefill id {self.prefill_node_id} device_index {self.device_index} thread quit")
        return

    def __del__(self):
        self.has_error = True
        self.clear_tasks()

        # 强制关闭连接和杀掉传输进程
        if self.process is not None:
            logger.warning(f"trans kv process {self.process.pid} is killed")
            os.kill(self.process.pid, signal.SIGKILL)
        pass


class DecodeKVMoveManager(rpyc.Service):
    def __init__(self, args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
        super().__init__()
        self.args = args
        self.dp_size = args.dp
        self.info_queue = info_queue
        self.mem_queues = mem_queues
        self.infer_rpyc_lock = threading.Lock()
        self.infer_rpyc_objs: List[PDDecodeInferRpcServer] = []
        self.node_id_to_trans_obj: Dict[str, TransProcessObj] = {}
        for port in self.args.pd_tp_infer_rpyc_ports:
            socket_path = f"/tmp/decode_node_infer_rpyc_{port}"
            from rpyc.utils.factory import unix_connect

            con = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})
            self.infer_rpyc_objs.append(con.root)
            logger.info(f"rpyc connect to port: {port} ok")

        from .up_status import start_up_kv_status_process

        self.up_status_in_queue = mp.Queue()
        self.up_status_out_queue = mp.Queue()
        start_up_kv_status_process(self.args, self.up_status_in_queue, self.up_status_out_queue)

        # 需要每个卡有一个锁来规划每次只能有一个tran obj 操作对应显卡上的传输任务。
        self.device_locks = [threading.Lock() for _ in range(self.args.tp)]
        return

    async def wait_all_future_finish(self, futures: List[AsyncResult]):
        await asyncio.gather(*[asyncio.to_thread(future.wait) for future in futures])
        return

    def _alloc_to_frozen_some_tokens(self, task: KVMoveTask) -> Tuple[int, Optional[List[int]]]:
        if self.dp_size == 1:
            with self.infer_rpyc_lock:
                futures: List[AsyncResult] = []
                for conn in self.infer_rpyc_objs:
                    futures.append(rpyc.async_(conn.alloc_to_frozen_some_tokens)(task))
                asyncio.run(self.wait_all_future_finish(futures))
                return 0, obtain(futures[0].value)
        else:
            dp_indexes = list(range(self.dp_size))
            random.shuffle(dp_indexes)

            with self.infer_rpyc_lock:
                for dp_index in dp_indexes:
                    conn = self.infer_rpyc_objs[dp_index]
                    futures = [rpyc.async_(conn.alloc_to_frozen_some_tokens)(task)]
                    asyncio.run(self.wait_all_future_finish(futures))
                    ans_value = obtain(futures[0].value)
                    if ans_value is not None:
                        return dp_index, ans_value
            return None, None

    def _put_kv_received_to_radix_cache(self, task: KVMoveTask) -> None:
        if self.dp_size == 1:
            with self.infer_rpyc_lock:
                futures: List[AsyncResult] = []
                for conn in self.infer_rpyc_objs:
                    futures.append(rpyc.async_(conn.put_kv_received_to_radix_cache)(task.group_request_id))
                asyncio.run(self.wait_all_future_finish(futures))
        else:
            with self.infer_rpyc_lock:
                futures: List[AsyncResult] = []
                conn = self.infer_rpyc_objs[task.decode_dp_index]
                futures.append(rpyc.async_(conn.put_kv_received_to_radix_cache)(task.group_request_id))
                asyncio.run(self.wait_all_future_finish(futures))
        return

    def _fail_to_realese_forzen_tokens(self, task: KVMoveTask) -> None:
        if self.dp_size == 1:
            with self.infer_rpyc_lock:
                futures: List[AsyncResult] = []
                for conn in self.infer_rpyc_objs:
                    futures.append(rpyc.async_(conn.fail_to_realese_forzen_tokens)(task.group_request_id))
                asyncio.run(self.wait_all_future_finish(futures))
        else:
            with self.infer_rpyc_lock:
                futures: List[AsyncResult] = []
                conn = self.infer_rpyc_objs[task.decode_dp_index]
                futures.append(rpyc.async_(conn.fail_to_realese_forzen_tokens)(task.group_request_id))
                asyncio.run(self.wait_all_future_finish(futures))
        return

    def _unfrozen_time_out_reqs_tokens(self) -> None:
        # 这个接口比较特殊，可以不区分 dp 的具体模式
        with self.infer_rpyc_lock:
            futures: List[AsyncResult] = []
            for conn in self.infer_rpyc_objs:
                futures.append(rpyc.async_(conn.unfrozen_time_out_reqs_tokens)())
            asyncio.run(self.wait_all_future_finish(futures))
        return

    def _put_mem_manager_to_mem_queue(self) -> None:
        with self.infer_rpyc_lock:
            for obj in self.infer_rpyc_objs:
                obj.put_mem_manager_to_mem_queue()
        return

    def on_connect(self, conn):
        # 用于处理连接断开的时候，自动删除资源
        thread_local_data.prefill_node_id = None
        pass

    def on_disconnect(self, conn):
        # 用于处理连接断开的时候，自动删除资源
        if thread_local_data.prefill_node_id is not None:
            self.remove_trans_obj(thread_local_data.prefill_node_id)
            logger.info(f"prefill node id {thread_local_data.prefill_node_id} disconnect")
        pass

    def exposed_check_alive(self):
        # 用于 prefill node check 通信连接的状态。
        return

    def exposed_build_trans_process(self, prefill_node_id, nccl_ip, nccl_port):
        prefill_node_id, nccl_ip, nccl_port = list(map(obtain, [prefill_node_id, nccl_ip, nccl_port]))
        thread_local_data.prefill_node_id = prefill_node_id

        logger.info(f"build trans infos {prefill_node_id} {nccl_ip} {nccl_port}")
        # 如果有历史残留，一并移除
        self.remove_trans_obj(prefill_node_id)
        tran_obj = TransProcessObj()
        tran_obj.create(prefill_node_id, nccl_ip, nccl_port, self)
        self.node_id_to_trans_obj[prefill_node_id] = tran_obj
        return

    # 返回 None 代表繁忙， 放弃该任务的 kv 传送
    def exposed_request_data_transfer(self, tasks: List[KVMoveTask]) -> List[Optional[int]]:
        tasks = obtain(tasks)
        alloc_tokened_tasks = []
        ans_list = []
        try:
            for task in tasks:
                logger.info(f"exposed_request_data_transfer in {task.to_decode_log_info()}, type {type(task)}")

            trans_obj = self.get_trans_obj(tasks[0])
            assert trans_obj is not None

            for task in tasks:
                dp_index, decode_token_indexes = self._alloc_to_frozen_some_tokens(task)
                # 代表服务很繁忙，申请不到资源，需要拒绝
                if decode_token_indexes is None:
                    logger.info(f"req id {task.id()} request_data_transfer fail, server is busy")
                    ans_list.append(None)
                else:
                    task.decode_dp_index = dp_index
                    task.decode_token_indexes = decode_token_indexes
                    task.move_kv_len = len(decode_token_indexes)
                    ans_list.append(task.move_kv_len)
                    alloc_tokened_tasks.append(task)

        except BaseException as e:
            for task in alloc_tokened_tasks:
                self._fail_to_realese_forzen_tokens(task)
            self.remove_trans_obj(tasks[0].prefill_node_id)
            logger.exception(str(e))
            raise e

        try:
            trans_obj.put_list(alloc_tokened_tasks)
        except BaseException as e:
            logger.exception(str(e))
            for task in alloc_tokened_tasks:
                self._fail_to_realese_forzen_tokens(task)
            raise e

        return ans_list

    def get_next_device_index(self):
        counts = [0 for _ in range(self.args.tp)]
        for obj in self.node_id_to_trans_obj.values():
            counts[obj.device_index] += 1
        device_index = int(np.argmin(counts))
        return device_index

    def get_trans_obj(self, task: KVMoveTask):
        self.remove_dead_trans_obj()
        return self.node_id_to_trans_obj[task.prefill_node_id]

    def remove_dead_trans_obj(self):
        del_node_ids = []
        for node_id, t_obj in self.node_id_to_trans_obj.items():
            if t_obj.has_error or (not t_obj.thread.is_alive()):
                t_obj.has_error = True
                del_node_ids.append(node_id)

        for node_id in del_node_ids:
            self.node_id_to_trans_obj.pop(node_id, None)

        if len(del_node_ids) != 0:
            import gc

            gc.collect()
        return

    def remove_trans_obj(self, prefill_node_id):
        if prefill_node_id in self.node_id_to_trans_obj:
            trans_obj = self.node_id_to_trans_obj.pop(prefill_node_id, None)
            if trans_obj is not None:
                trans_obj.has_error = True
        return

    def timer_loop(self):
        while True:
            self._unfrozen_time_out_reqs_tokens()
            time.sleep(3.5)


def _init_env(args, info_queue: mp.Queue, mem_queues: List[mp.Queue], event: mp.Event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    manager = DecodeKVMoveManager(args, info_queue, mem_queues)
    t = ThreadedServer(manager, port=args.pd_decode_rpyc_port, protocol_config={"allow_pickle": True})
    threading.Thread(target=lambda: t.start(), daemon=True).start()

    event.set()

    manager.timer_loop()
    return


def start_decode_kv_move_manager_process(args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move manager process started")
    return

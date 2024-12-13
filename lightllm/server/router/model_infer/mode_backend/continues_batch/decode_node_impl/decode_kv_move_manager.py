import rpyc
import random
import asyncio
import sys
import os
import signal
import collections
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

        self.ready_task_queue_lock = threading.Lock()
        self.ready_task_queue = []

        # 启动处理任务的线程
        self.thread = threading.Thread(target=self.handle_loop, daemon=True)
        self.thread.start()

        self.put_to_radix_thread = threading.Thread(target=self.ready_handle_loop, daemon=True)
        self.put_to_radix_thread.start()
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
            if len(self.move_task_queue) != 0:
                for task in self.move_task_queue:
                    self.manager.put_to_fail_release_task_queue(task)
            self.move_task_queue = []

    def put_ready(self, move_task):
        if self.has_error:
            raise Exception(f"trans obj {self.prefill_node_id} has error, can not put")
        with self.ready_task_queue_lock:
            self.ready_task_queue.append(move_task)

    def get_ready_tasks(self):
        with self.ready_task_queue_lock:
            move_tasks: List[KVMoveTask] = self.ready_task_queue[0:KV_MOVE_MAX_NUM]
            self.ready_task_queue = self.ready_task_queue[KV_MOVE_MAX_NUM:]
        return move_tasks

    def clear_ready_tasks(self):
        with self.ready_task_queue_lock:
            if len(self.ready_task_queue) != 0:
                for task in self.ready_task_queue:
                    self.manager.put_to_fail_release_task_queue(task)
            self.ready_task_queue = []

    def handle_loop(self):
        while not self.has_error:
            move_tasks: List[KVMoveTask] = self.get_tasks()
            if len(move_tasks) == 0:
                time.sleep(0.01)
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
                        self.put_ready(task)
                        move_tasks.remove(task)

            except BaseException as e:
                logger.exception(str(e))
                self.has_error = True
                self.clear_tasks()
                self.manager.remove_trans_obj(self.prefill_node_id)

            finally:
                for move_task in move_tasks:
                    self.manager.put_to_fail_release_task_queue(move_task)

        logger.error(f"trans thread, prefill id {self.prefill_node_id} device_index {self.device_index} thread quit")
        return

    def ready_handle_loop(self):
        while not self.has_error:
            move_tasks: List[KVMoveTask] = self.get_ready_tasks()
            if len(move_tasks) == 0:
                time.sleep(0.01)
                continue

            for task in move_tasks:
                logger.info(f"deocode node get put radix task {task.to_decode_log_info()}")

            try:
                # random to check stats
                if random.randint(0, 20) == 10:
                    self.check_trans_process()

                self.manager._put_kv_received_to_radix_cache(move_tasks)
                iter_move_tasks = move_tasks.copy()
                move_tasks.clear()

                for task in iter_move_tasks:
                    logger.info(f"decode node put kv to radix cache ok, req_id: {task.id()}")
                    self.manager.up_status_in_queue.put(
                        UpKVStatus(group_request_id=task.group_request_id, dp_index=task.decode_dp_index)
                    )
                    logger.info(f"decode node up kv status req_id: {task.id()} finished")

            except BaseException as e:
                logger.exception(str(e))
                self.has_error = True
                self.clear_ready_tasks()
                self.manager.remove_trans_obj(self.prefill_node_id)

            finally:
                for move_task in move_tasks:
                    self.manager.put_to_fail_release_task_queue(move_task)

        logger.error(
            f"put to radix thread, prefill id {self.prefill_node_id} device_index {self.device_index} thread quit"
        )
        return

    def __del__(self):
        self.has_error = True
        self.clear_tasks()
        self.clear_ready_tasks()

        logger.error(f"trans obj deled, prefill node id {self.prefill_node_id} device_index {self.device_index}")

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

        # fail release queue
        self.fail_to_release_queue_lock = threading.Lock()
        self.fail_to_release_queue = []
        self.fail_to_release_thread = threading.Thread(target=self.handle_fail_release_task_loop, daemon=True)
        self.fail_to_release_thread.start()

        # 需要每个卡有一个锁来规划每次只能有一个tran obj 操作对应显卡上的传输任务。
        self.device_locks = [threading.Lock() for _ in range(self.args.tp)]
        return

    def put_to_fail_release_task_queue(self, task: KVMoveTask):
        with self.fail_to_release_queue_lock:
            self.fail_to_release_queue.append(task)

    def get_fail_release_tasks_from_queue(self):
        with self.fail_to_release_queue_lock:
            ans = self.fail_to_release_queue[0:KV_MOVE_MAX_NUM]
            self.fail_to_release_queue = self.fail_to_release_queue[KV_MOVE_MAX_NUM:]
        return ans

    def handle_fail_release_task_loop(self):
        while True:
            handle_list: List[KVMoveTask] = self.get_fail_release_tasks_from_queue()
            if len(handle_list) == 0:
                time.sleep(0.01)
            else:
                self._fail_to_realese_forzen_tokens(handle_list)
        return

    async def wait_all_future_finish(self, futures: List[AsyncResult]):
        await asyncio.gather(*[asyncio.to_thread(future.wait) for future in futures])
        return

    def _tp_alloc_to_frozen_some_tokens(self, tasks: List[KVMoveTask]) -> List[Optional[List[int]]]:
        assert self.dp_size == 1
        with self.infer_rpyc_lock:
            futures: List[AsyncResult] = []
            for conn in self.infer_rpyc_objs:
                futures.append(rpyc.async_(conn.alloc_to_frozen_some_tokens)(tasks))
            asyncio.run(self.wait_all_future_finish(futures))
            return obtain(futures[0].value)

    def _dp_alloc_to_frozen_some_tokens(self, dp_tasks: List[List[KVMoveTask]]) -> List[List[Optional[List[int]]]]:
        assert self.dp_size != 1
        with self.infer_rpyc_lock:
            futures = []
            for dp_index in range(self.dp_size):
                conn = self.infer_rpyc_objs[dp_index]
                futures.append(rpyc.async_(conn.alloc_to_frozen_some_tokens)(dp_tasks[dp_index]))
            asyncio.run(self.wait_all_future_finish(futures))
            ans_values = [obtain(futures[dp_index].value) for dp_index in range(self.dp_size)]
            return ans_values

    def _put_kv_received_to_radix_cache(self, tasks: List[KVMoveTask]) -> None:
        if self.dp_size == 1:
            with self.infer_rpyc_lock:
                futures: List[AsyncResult] = []
                for conn in self.infer_rpyc_objs:
                    futures.append(
                        rpyc.async_(conn.put_kv_received_to_radix_cache)([task.group_request_id for task in tasks])
                    )
                asyncio.run(self.wait_all_future_finish(futures))
        else:
            with self.infer_rpyc_lock:
                dp_to_tasks = collections.defaultdict(list)
                for task in tasks:
                    dp_to_tasks[task.decode_dp_index].append(task)
                futures: List[AsyncResult] = []
                for decode_dp_index, _tasks in dp_to_tasks.items():
                    conn = self.infer_rpyc_objs[decode_dp_index]
                    futures.append(
                        rpyc.async_(conn.put_kv_received_to_radix_cache)([task.group_request_id for task in _tasks])
                    )
                asyncio.run(self.wait_all_future_finish(futures))
        return

    def _fail_to_realese_forzen_tokens(self, tasks: List[KVMoveTask]) -> None:
        if self.dp_size == 1:
            with self.infer_rpyc_lock:
                futures: List[AsyncResult] = []
                for conn in self.infer_rpyc_objs:
                    futures.append(
                        rpyc.async_(conn.fail_to_realese_forzen_tokens)([task.group_request_id for task in tasks])
                    )
                asyncio.run(self.wait_all_future_finish(futures))
        else:
            with self.infer_rpyc_lock:
                dp_to_tasks = collections.defaultdict(list)
                for task in tasks:
                    dp_to_tasks[task.decode_dp_index].append(task)
                futures: List[AsyncResult] = []
                for decode_dp_index, _tasks in dp_to_tasks.items():
                    conn = self.infer_rpyc_objs[decode_dp_index]
                    futures.append(
                        rpyc.async_(conn.fail_to_realese_forzen_tokens)([task.group_request_id for task in _tasks])
                    )
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
        tasks: List[KVMoveTask] = obtain(tasks)
        alloc_tokened_tasks = []
        ans_list = []
        try:
            for task in tasks:
                logger.info(f"exposed_request_data_transfer in {task.to_decode_log_info()}, type {type(task)}")

            trans_obj = self.get_trans_obj(tasks[0])
            assert trans_obj is not None

            if self.dp_size == 1:
                list_decode_token_indexes = self._tp_alloc_to_frozen_some_tokens(tasks)
                for i, task in enumerate(tasks):
                    decode_token_indexes = list_decode_token_indexes[i]
                    if decode_token_indexes is None:
                        logger.info(f"req id {task.id()} request_data_transfer fail, server is busy")
                        ans_list.append(None)
                    else:
                        task.decode_dp_index = 0
                        task.decode_token_indexes = decode_token_indexes
                        task.move_kv_len = len(decode_token_indexes)
                        ans_list.append(task.move_kv_len)
                        alloc_tokened_tasks.append(task)
            else:
                id_to_test_range = {task.group_request_id: random.shuffle(list(range(self.dp_size))) for task in tasks}
                id_has_result = {}
                for test_index in range(self.dp_size):
                    dp_tasks = [[] for _ in range(self.dp_size)]
                    for task in tasks:
                        if task.group_request_id not in id_has_result:
                            test_dp_index = id_to_test_range[task.group_request_id][test_index]
                            dp_tasks[test_dp_index].append(task)
                    if not all(len(t) == 0 for t in dp_tasks):
                        dp_tasks_ans = self._dp_alloc_to_frozen_some_tokens(dp_tasks)
                        for dp_index in range(self.dp_size):
                            for task, decode_token_indexes in zip(dp_tasks[dp_index], dp_tasks_ans[dp_index]):
                                if decode_token_indexes is not None:
                                    id_has_result[task.group_request_id] = (dp_index, decode_token_indexes)
                for task in tasks:
                    if task.group_request_id in id_has_result:
                        task.decode_dp_index = id_has_result[task.group_request_id][0]
                        task.decode_token_indexes = id_has_result[task.group_request_id][1]
                        task.move_kv_len = len(task.decode_token_indexes)
                        ans_list.append(task.move_kv_len)
                        alloc_tokened_tasks.append(task)
                    else:
                        logger.info(f"req id {task.id()} request_data_transfer fail, server is busy")
                        ans_list.append(None)

        except BaseException as e:
            if len(alloc_tokened_tasks) != 0:
                for task in alloc_tokened_tasks:
                    self.put_to_fail_release_task_queue(task)
                alloc_tokened_tasks = []

            self.remove_trans_obj(tasks[0].prefill_node_id)
            logger.exception(str(e))
            raise e

        try:
            trans_obj.put_list(alloc_tokened_tasks)
        except BaseException as e:
            logger.exception(str(e))
            if len(alloc_tokened_tasks) != 0:
                for task in alloc_tokened_tasks:
                    self.put_to_fail_release_task_queue(task)
                alloc_tokened_tasks = []
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

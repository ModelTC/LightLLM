import rpyc
import random
import asyncio
import os
import signal
import collections
import time
import psutil
import threading
import inspect
from rpyc.utils.classic import obtain
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from rpyc import ThreadedServer
from lightllm.utils.log_utils import init_logger
from .decode_infer_rpyc import PDDecodeInferRpcServer
from ..task_queue import TaskQueue
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask, UpKVStatus
from lightllm.utils.retry_utils import retry
import numpy as np
from rpyc import AsyncResult
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry

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
    manager: "DecodeKVMoveManager" = None
    has_error: bool = False
    ready_to_move_queue: TaskQueue = None
    kv_move_thread: threading.Thread = None
    move_finished_queue: TaskQueue = None
    put_to_radix_thread: threading.Thread = None
    latest_check_time: float = None

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
        self.latest_check_time = time.time()

        self.ready_to_move_queue = TaskQueue(
            get_func=lambda datas: datas[0:1], fail_func=self.manager.put_to_fail_release_task_queue
        )
        self.kv_move_thread = threading.Thread(target=self.kv_move_loop, daemon=True)
        self.kv_move_thread.start()

        self.move_finished_queue = TaskQueue(
            get_func=lambda datas: datas[0:KV_MOVE_MAX_NUM], fail_func=self.manager.put_to_fail_release_task_queue
        )
        self.put_to_radix_thread = threading.Thread(target=self.put_to_radix_loop, daemon=True)
        self.put_to_radix_thread.start()
        return

    def check_trans_process(self, raise_exception=True):
        process = psutil.Process(self.process.pid)
        if not (process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
            self.set_has_error()
            if raise_exception:
                raise Exception(f"trans process: {self.process.pid} is dead")
        return

    def timer_to_check_status(self, raise_exception=True):
        if time.time() - self.latest_check_time >= 2.0:
            self.latest_check_time = time.time()
            self.check_trans_process(raise_exception=raise_exception)
        return

    def _transfer_kv(self, move_tasks: List[KVMoveTask]):
        with self.manager.device_locks[self.device_index]:
            self.task_in_queue.put(move_tasks.copy(), timeout=10)
            assert self.task_out_queue.get(timeout=60) == "ok"
            logger.info(f"_transfer_kv ok {move_tasks[0].to_decode_log_info()}")

            # 标记 decode 接收到 kv cache 的时间
            for move_task in move_tasks:
                move_task.mark_start_time = time.time()

            self.move_finished_queue.put_list(move_tasks)
            move_tasks.clear()

    def kv_move_loop(self):
        func_name = self.kv_move_loop.__name__
        while not self.has_error:
            move_tasks: List[List[KVMoveTask]] = self.ready_to_move_queue.get_tasks(log_tag="ready_to_move_queue")
            if len(move_tasks) == 0:
                time.sleep(0.01)
                continue

            if len(move_tasks) != 1:
                logger.error(f"error get need 1, but get {len(move_tasks)}")
                assert False

            move_tasks = move_tasks[0]
            for task in move_tasks:
                logger.info(f"{func_name} get task {task.to_decode_log_info()}")

            try:
                self.timer_to_check_status(raise_exception=True)

                if not kv_trans_use_p2p():
                    with self.manager.kv_trans_lock:
                        self._transfer_kv(move_tasks)
                else:
                    self._transfer_kv(move_tasks)

            except BaseException as e:
                logger.exception(str(e))
                self.set_has_error()
                self.ready_to_move_queue.clear_tasks()
                self.manager.remove_trans_obj(self.prefill_node_id)

            finally:
                self.manager.put_to_fail_release_task_queue(move_tasks)

        logger.error(f"{func_name} prefill id {self.prefill_node_id} device_index {self.device_index} thread quit")
        return

    def put_to_radix_loop(self):
        func_name = self.put_to_radix_loop.__name__
        while not self.has_error:
            move_tasks: List[KVMoveTask] = self.move_finished_queue.get_tasks(log_tag="move_finished_queue")
            if len(move_tasks) == 0:
                time.sleep(0.01)
                continue

            for task in move_tasks:
                logger.info(f"{func_name} get put radix task {task.to_decode_log_info()}")

            try:
                # random to check stats
                self.timer_to_check_status(raise_exception=True)

                self.manager._put_kv_received_to_radix_cache(move_tasks.copy())
                for task in move_tasks.copy():
                    logger.info(
                        f"{func_name} put kv to radix cache ok, req_id: {task.id()} cost_time {task.get_cost_time()} s"
                    )
                    self.manager.up_status_in_queue.put(
                        UpKVStatus(group_request_id=task.group_request_id, dp_index=task.decode_dp_index)
                    )
                    logger.info(f"{func_name} up kv status req_id: {task.id()} finished")
                move_tasks.clear()

            except BaseException as e:
                logger.exception(str(e))
                self.set_has_error()
                self.move_finished_queue.clear_tasks()
                self.manager.remove_trans_obj(self.prefill_node_id)

            finally:
                self.manager.put_to_fail_release_task_queue(move_tasks)

        logger.error(f"{func_name}, prefill id {self.prefill_node_id} device_index {self.device_index} thread quit")
        return

    def wait_thread_quit(self):
        if self.kv_move_thread is not None:
            if self.kv_move_thread.is_alive():
                try:
                    self.kv_move_thread.join()
                except:
                    pass
        if self.put_to_radix_thread is not None:
            if self.put_to_radix_thread.is_alive():
                try:
                    self.put_to_radix_thread.join()
                except:
                    pass
        return

    def has_error_status(self):
        try:
            assert self.has_error is False
            assert self.kv_move_thread.is_alive()
            assert self.put_to_radix_thread.is_alive()
        except BaseException as e:
            logger.exception(str(e))
            self.set_has_error()
            return True

        return False

    def set_has_error(self):
        self.has_error = True
        try:
            self.ready_to_move_queue.has_error = True
            self.move_finished_queue.has_error = True
        except:
            pass
        return

    def __del__(self):
        logger.error(f"trans obj del start, prefill node id {self.prefill_node_id} device_index {self.device_index}")

        try:
            self.set_has_error()
            self.wait_thread_quit()
            if self.ready_to_move_queue is not None:
                self.ready_to_move_queue.clear_tasks()
            if self.move_finished_queue is not None:
                self.move_finished_queue.clear_tasks()
        except BaseException as e:
            logger.exception(str(e))

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
        self.fail_to_release_queue = TaskQueue(get_func=lambda datas: datas[0:KV_MOVE_MAX_NUM], fail_func=None)
        self.fail_to_release_thread = threading.Thread(target=self.handle_fail_release_task_loop, daemon=True)
        self.fail_to_release_thread.start()

        self.kv_trans_lock = threading.Lock()
        # 需要每个卡有一个锁来规划每次只能有一个tran obj 操作对应显卡上的传输任务。
        self.device_locks = [threading.Lock() for _ in range(self.args.tp)]
        return

    def put_to_fail_release_task_queue(self, task: Union[KVMoveTask, List[KVMoveTask]]):
        if isinstance(task, KVMoveTask):
            self.fail_to_release_queue.put(task)
        elif isinstance(task, list):
            self.fail_to_release_queue.put_list(task)
        else:
            assert False, "error input"
        return

    def handle_fail_release_task_loop(self):
        while True:
            handle_list: List[KVMoveTask] = self.fail_to_release_queue.get_tasks(log_tag="fail_to_release_queue")
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
            import gc

            gc.collect()
        pass

    def exposed_check_alive(self):
        # 用于 prefill node check 通信连接的状态。
        return

    def exposed_build_trans_process(self, prefill_node_id, nccl_ip, nccl_port, prefill_node_max_kv_trans_num):
        prefill_node_id, nccl_ip, nccl_port, prefill_node_max_kv_trans_num = list(
            map(obtain, [prefill_node_id, nccl_ip, nccl_port, prefill_node_max_kv_trans_num])
        )
        thread_local_data.prefill_node_id = prefill_node_id

        logger.info(f"build trans infos {prefill_node_id} {nccl_ip} {nccl_port}")
        # 如果有历史残留，一并移除
        self.remove_trans_obj(prefill_node_id)
        tran_obj = TransProcessObj()
        tran_obj.create(prefill_node_id, nccl_ip, nccl_port, self)
        self.node_id_to_trans_obj[prefill_node_id] = tran_obj
        return min(prefill_node_max_kv_trans_num, self.args.max_total_token_num)

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
            self.put_to_fail_release_task_queue(alloc_tokened_tasks)
            alloc_tokened_tasks = []
            self.remove_trans_obj(tasks[0].prefill_node_id)
            logger.exception(str(e))
            raise e

        try:
            if len(alloc_tokened_tasks) != 0:
                trans_obj.ready_to_move_queue.put(alloc_tokened_tasks)
        except BaseException as e:
            logger.exception(str(e))
            self.put_to_fail_release_task_queue(alloc_tokened_tasks)
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
            if t_obj.has_error_status():
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
                trans_obj.set_has_error()
        return

    def timer_loop(self):
        while True:
            self._unfrozen_time_out_reqs_tokens()
            time.sleep(3.5)


def _init_env(args, info_queue: mp.Queue, mem_queues: List[mp.Queue], event: mp.Event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
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
    logger.info("decode kv move manager process started")
    return

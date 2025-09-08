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
import setproctitle
from rpyc.utils.classic import obtain
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from rpyc import ThreadedServer
from lightllm.utils.log_utils import init_logger
from .decode_infer_rpyc import PDDecodeInferRpcServer
from ..task_queue import TaskQueue
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask, UpKVStatus, PDTransJoinInfo, PDTransLeaveInfo
from lightllm.utils.retry_utils import retry
import numpy as np
from rpyc import AsyncResult
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_unique_server_name

logger = init_logger(__name__)

thread_local_data = threading.local()

KV_MOVE_MAX_NUM = 16


class DecodeKVMoveManager(rpyc.Service):
    def __init__(self, args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
        super().__init__()
        self.args = args
        # args.dp // args.nnodes 在跨机tp的场景下，可能为0
        self.dp_size_in_node = max(1, args.dp // args.nnodes)
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = args.tp // args.dp
        # 不支持跨机tp的pd 分离策略
        assert self.dp_world_size <= self.node_world_size

        self.info_queue = info_queue
        self.mem_queues = mem_queues
        self.infer_rpyc_lock = threading.Lock()
        self.infer_rpyc_objs: List[PDDecodeInferRpcServer] = []

        from .decode_trans_obj import KVTransConnectObj

        self.connect_id_to_trans_obj: Dict[str, KVTransConnectObj] = {}
        for port in self.args.pd_node_infer_rpyc_ports:
            socket_path = f"/tmp/{get_unique_server_name()}_decode_node_infer_rpyc_{port}"
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

        # 在不使用p2p 复制kv 的方案时，需要全局的传输锁进行控制。这个时候kv传输的效率会下降。
        self.kv_trans_lock = threading.Lock()

        from .decode_trans_obj import KVTransProcess

        self.kv_trans_processes: List[KVTransProcess] = [None] * self.node_world_size
        for device_id in range(self.node_world_size):
            self.kv_trans_processes[device_id] = KVTransProcess()
            assert self.kv_trans_processes[device_id].init_all(device_id, self)

        return

    # ==================================================================================
    # _dp_alloc_to_frozen_some_tokens
    # _put_kv_received_to_radix_cache
    # _fail_to_realese_forzen_tokens
    # _unfrozen_time_out_reqs_tokens
    # _put_mem_manager_to_mem_queue
    # 上述接口都是 kv move manager 与推理进程进行交互的接口，主要用于申请锁定kv资源或者释放
    # kv资源的接口
    # ==================================================================================

    async def wait_all_future_finish(self, futures: List[AsyncResult]):
        await asyncio.gather(*[asyncio.to_thread(future.wait) for future in futures])
        return

    def _dp_alloc_to_frozen_some_tokens(self, dp_tasks: List[List[KVMoveTask]]) -> List[List[Optional[List[int]]]]:
        with self.infer_rpyc_lock:
            futures = []
            for dp_index in range(self.dp_size_in_node):
                conn_start = dp_index * self.dp_world_size
                conn_end = (dp_index + 1) * self.dp_world_size
                conns = self.infer_rpyc_objs[conn_start:conn_end]
                for conn in conns:
                    futures.append(rpyc.async_(conn.alloc_to_frozen_some_tokens)(dp_tasks[dp_index]))

            asyncio.run(self.wait_all_future_finish(futures))
            ans_values = [
                obtain(futures[dp_index * self.dp_world_size].value) for dp_index in range(self.dp_size_in_node)
            ]
            return ans_values

    def _put_kv_received_to_radix_cache(self, tasks: List[KVMoveTask]) -> None:
        with self.infer_rpyc_lock:
            dp_to_tasks = collections.defaultdict(list)
            for task in tasks:
                dp_to_tasks[task.decode_dp_index].append(task)
            futures: List[AsyncResult] = []
            for decode_dp_index, _tasks in dp_to_tasks.items():
                conn_start = decode_dp_index * self.dp_world_size
                conn_end = (decode_dp_index + 1) * self.dp_world_size
                conns = self.infer_rpyc_objs[conn_start:conn_end]
                for conn in conns:
                    futures.append(
                        rpyc.async_(conn.put_kv_received_to_radix_cache)([task.group_request_id for task in _tasks])
                    )
            asyncio.run(self.wait_all_future_finish(futures))
        return

    def _fail_to_realese_forzen_tokens(self, tasks: List[KVMoveTask]) -> None:
        with self.infer_rpyc_lock:
            dp_to_tasks = collections.defaultdict(list)
            for task in tasks:
                dp_to_tasks[task.decode_dp_index].append(task)
            futures: List[AsyncResult] = []
            for decode_dp_index, _tasks in dp_to_tasks.items():
                conn_start = decode_dp_index * self.dp_world_size
                conn_end = (decode_dp_index + 1) * self.dp_world_size
                conns = self.infer_rpyc_objs[conn_start:conn_end]
                for conn in conns:
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

    # ==================================================================================
    # put_to_fail_release_task_queue 将因为一些原因失败，需要释放锁定的kv资源的请求放入到
    # 对应的处理队列中，handle_fail_release_task_loop 是一个循环的线程，专门处理这些失败的请求
    # 通过调用与推理进程交互的接口，释放掉申请锁定的 kv 资源。
    # ==================================================================================

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

    # ==================================================================================
    # on_connect
    # on_disconnect
    # exposed_check_alive
    # exposed_build_trans_process
    # exposed_request_data_transfer
    # 上述接口是decode kv move manager 暴露的 rpyc 调用接口，用于 prefill kv move manager
    # 进行连接，进行一些元数据资源的交互。
    # ==================================================================================

    def on_connect(self, conn):
        # 用于处理连接断开的时候，自动删除资源
        thread_local_data.connect_id = None
        pass

    def on_disconnect(self, conn):
        # 用于处理连接断开的时候，自动删除资源
        if thread_local_data.connect_id is not None:
            self.remove_trans_obj(thread_local_data.connect_id)
            logger.info(f"connect id {thread_local_data.connect_id} disconnect")
            import gc

            gc.collect()
        pass

    def exposed_check_alive(self):
        # 用于 prefill node check 通信连接的状态。
        return

    def exposed_build_trans_connect(
        self, prefill_node_id, pd_prefill_nccl_ip, pd_prefill_nccl_port, prefill_node_max_kv_trans_num, connect_id
    ):
        prefill_node_id, pd_prefill_nccl_ip, pd_prefill_nccl_port, prefill_node_max_kv_trans_num = list(
            map(obtain, [prefill_node_id, pd_prefill_nccl_ip, pd_prefill_nccl_port, prefill_node_max_kv_trans_num])
        )
        connect_id = obtain(connect_id)
        thread_local_data.connect_id = connect_id

        logger.info(f"build trans infos {prefill_node_id} {pd_prefill_nccl_ip} {pd_prefill_nccl_port} {connect_id}")

        from .decode_trans_obj import KVTransConnectObj

        tran_obj = KVTransConnectObj()
        tran_obj.create(connect_id, prefill_node_id, pd_prefill_nccl_ip, pd_prefill_nccl_port, self)
        self.connect_id_to_trans_obj[connect_id] = tran_obj
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

            id_to_test_range = {}
            for task in tasks:
                test_dp_indexes = list(range(self.dp_size_in_node))
                random.shuffle(test_dp_indexes)
                id_to_test_range[task.group_request_id] = test_dp_indexes

            id_has_result = {}
            for test_index in range(self.dp_size_in_node):
                dp_tasks = [[] for _ in range(self.dp_size_in_node)]
                for task in tasks:
                    if task.group_request_id not in id_has_result:
                        test_dp_index = id_to_test_range[task.group_request_id][test_index]
                        dp_tasks[test_dp_index].append(task)
                if not all(len(t) == 0 for t in dp_tasks):
                    dp_tasks_ans = self._dp_alloc_to_frozen_some_tokens(dp_tasks)
                    for dp_index in range(self.dp_size_in_node):
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
            self.remove_trans_obj(tasks[0].connect_id)
            logger.exception(str(e))
            raise e

        if alloc_tokened_tasks:
            trans_obj.ready_to_move_queue.put(
                alloc_tokened_tasks, error_handle_func=self.put_to_fail_release_task_queue
            )

        return ans_list

    # ==================================================================================
    # 定时检测kv 传输成功，但是长时间没有pd master来触发推理的请求，
    # 释放这些超时请求占用的kv资源
    # ==================================================================================

    def timer_loop(self):
        try:
            while True:
                self._unfrozen_time_out_reqs_tokens()
                time.sleep(3.5)
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e

    # ==================================================================================
    # 定时检测传输进程的健康状态，出现问题拉崩整个系统触发重启
    # ==================================================================================

    def check_trans_process_loop(self):
        try:
            while True:
                for device_id in range(self.node_world_size):
                    if not self.kv_trans_processes[device_id].is_trans_process_health():
                        raise Exception(f"device_id {device_id} kv process is unhealth")

                time.sleep(10.0)
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))

            for device_id in range(self.node_world_size):
                self.kv_trans_processes[device_id].killself()

            # 杀掉当前进程的父进程（router), 触发全局崩溃
            os.kill(os.getppid(), signal.SIGKILL)
            os.kill(os.getpid(), signal.SIGKILL)
            raise e

    # ==================================================================================
    # 常用辅助功能函数
    # ==================================================================================
    def get_next_device_index(self):
        counts = [0 for _ in range(self.node_world_size)]
        for obj in self.connect_id_to_trans_obj.values():
            counts[obj.device_index] += 1
        device_index = int(np.argmin(counts))
        return device_index

    def get_trans_obj(self, task: KVMoveTask):
        self.__remove_dead_trans_obj()
        return self.connect_id_to_trans_obj[task.connect_id]

    def __remove_dead_trans_obj(self):
        del_connect_ids = []
        for connect_id, t_obj in self.connect_id_to_trans_obj.items():
            if t_obj.has_error_status():
                del_connect_ids.append(connect_id)

        for connect_id in del_connect_ids:
            self.connect_id_to_trans_obj.pop(connect_id, None)

        if del_connect_ids:
            import gc

            gc.collect()
        return

    def remove_trans_obj(self, connect_id):
        if connect_id in self.connect_id_to_trans_obj:
            trans_obj = self.connect_id_to_trans_obj.pop(connect_id, None)
            if trans_obj is not None:
                trans_obj.set_has_error()
        return


def _init_env(args, info_queue: mp.Queue, mem_queues: List[mp.Queue], event: mp.Event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::decode_kv_move_manager")

    manager = DecodeKVMoveManager(args, info_queue, mem_queues)
    t = ThreadedServer(manager, port=args.pd_decode_rpyc_port, protocol_config={"allow_pickle": True})
    threading.Thread(target=lambda: t.start(), daemon=True).start()

    kv_trans_process_check = threading.Thread(target=manager.check_trans_process_loop, daemon=True)
    kv_trans_process_check.start()

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

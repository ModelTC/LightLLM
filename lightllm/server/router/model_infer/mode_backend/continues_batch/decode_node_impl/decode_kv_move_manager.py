import rpyc
import asyncio
import sys
import os
import signal
import time
import psutil
import threading
import random
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
from ..prefill_node_impl.prefill_kv_move_manager import DecodeBusyError

logger = init_logger(__name__)

thread_local_data = threading.local()


@dataclass
class TransProcessObj:
    prefill_node_id: str = None
    process: mp.Process = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    nccl_ip: str = None
    nccl_port: str = None
    device_index: int = None

    def create(self, prefill_node_id: str, nccl_ip: str, nccl_port: int, manager: "DecodeKVMoveManager"):
        from .decode_trans_process import start_decode_trans_process

        task_in_queue = mp.Queue()
        task_out_queue = mp.Queue()
        device_index = manager.get_next_device_index()
        proc = start_decode_trans_process(
            manager.args, device_index, nccl_ip, nccl_port, task_in_queue, task_out_queue, manager.mem_queues
        )
        assert task_out_queue.get(timeout=30) == "proc_start"
        with manager.infer_rpyc_lock:
            for obj in manager.infer_rpyc_objs:
                obj.put_mem_manager_to_mem_queue()
        assert task_out_queue.get(timeout=60) == "get_mem_managers_ok"
        assert task_out_queue.get(timeout=60) == "nccl_ok"

        self.prefill_node_id = prefill_node_id
        self.process = proc
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.nccl_ip = nccl_ip
        self.nccl_port = nccl_port
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

        # 开启tp个线程和队列来处理,每个队列处理一张卡上的任务
        self.task_queues = [queue.Queue() for _ in range(self.args.tp)]
        for i in range(self.args.tp):
            threading.Thread(target=self.handle_loop, args=(self.task_queues[i],), daemon=True).start()
        threading.Thread(target=self.timer_loop, daemon=True).start()
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

    def on_connect(self, conn):
        # 用于处理连接断开的时候，自动删除资源
        thread_local_data.prefill_node_id = None
        pass

    def on_disconnect(self, conn):
        # 用于处理连接断开的时候，自动删除资源
        if thread_local_data.prefill_node_id is not None:
            self.node_id_to_trans_obj.pop(thread_local_data.prefill_node_id, None)
            logger.info(f"prefill node id {thread_local_data.prefill_node_id} disconnect")
        pass

    def exposed_check_alive(self):
        # 用于 prefill node check 通信连接的状态。
        return

    def exposed_build_trans_process(self, prefill_node_id, nccl_ip, nccl_port):
        prefill_node_id, nccl_ip, nccl_port = list(map(obtain, [prefill_node_id, nccl_ip, nccl_port]))
        thread_local_data.prefill_node_id = prefill_node_id

        logger.info(f"build trans infos {prefill_node_id} {nccl_ip} {nccl_port}")
        if prefill_node_id in self.node_id_to_trans_obj:
            self.node_id_to_trans_obj.pop(prefill_node_id, None)
        tran_obj = TransProcessObj()
        tran_obj.create(prefill_node_id, nccl_ip, nccl_port, self)
        self.node_id_to_trans_obj[prefill_node_id] = tran_obj
        return

    # 返回 None 代表繁忙， 放弃该任务的 kv 传送
    def exposed_request_data_transfer(self, task: KVMoveTask) -> Optional[int]:
        task = obtain(task)
        logger.info(f"exposed_request_data_transfer in {task.to_decode_log_info()}")
        try:
            trans_obj = self.get_trans_obj(task)
            device_index = trans_obj.device_index
            assert trans_obj is not None

            dp_index, decode_token_indexes = self._alloc_to_frozen_some_tokens(task)
            # 代表服务很繁忙，申请不到资源，需要拒绝
            if decode_token_indexes is None:
                raise DecodeBusyError("token is full, busy")

            task.decode_dp_index = dp_index
            task.decode_token_indexes = decode_token_indexes
            task.move_kv_len = len(decode_token_indexes)

        except DecodeBusyError as e:
            logger.error(str(e))
            return None

        except BaseException as e:
            # 移除通信对象
            self.node_id_to_trans_obj.pop(task.prefill_node_id, None)
            trans_obj = None
            logger.exception(str(e))
            raise e

        self.task_queues[device_index].put(task)
        return task.move_kv_len

    def get_next_device_index(self):
        counts = [0 for _ in range(self.args.tp)]
        for obj in self.node_id_to_trans_obj.values():
            counts[obj.device_index] += 1
        device_index = int(np.argmin(counts))
        return device_index

    def get_trans_obj(self, task: KVMoveTask):
        return self.node_id_to_trans_obj[task.prefill_node_id]

    def handle_loop(self, task_queue: queue.Queue):
        try:
            while True:
                task = task_queue.get()
                if not isinstance(task, KVMoveTask):
                    logger.error("receive task type is not KVMoveTask")
                    sys.exit(-1)

                logger.info(f"deocode node get task {task.to_decode_log_info()}")
                try:
                    trans_obj = self.get_trans_obj(task)
                    trans_obj.task_in_queue.put(task, timeout=10)
                    assert trans_obj.task_out_queue.get(timeout=30) == "ok"
                    logger.info(f"deocode node transfer kv ok {task.to_decode_log_info()}")
                    # 成功了将 token 放入prompt cache中
                    self._put_kv_received_to_radix_cache(task)

                    logger.info(f"decode node put kv to radix cache ok, req_id: {task.id()}")
                    self.up_status_in_queue.put(
                        UpKVStatus(group_request_id=task.group_request_id, dp_index=task.decode_dp_index)
                    )
                    logger.info("decode node up kv status finished")
                except BaseException as e:
                    logger.exception(str(e))
                    # 失败了也需要释放锁定的 token
                    self._fail_to_realese_forzen_tokens(task)
                    logger.error(f"decode kv move task {task.to_decode_log_info()} has error, remove the trans_obj")
                    self.node_id_to_trans_obj.pop(task.prefill_node_id, None)
                finally:
                    # 去除引用否则进程无法自动退出
                    trans_obj = None
        except BaseException as e:
            logger.exception(str(e))
            raise e

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

    # 进入主循环
    while True:
        time.sleep(10)
    return


def start_decode_kv_move_manager_process(args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move manager process started")
    return

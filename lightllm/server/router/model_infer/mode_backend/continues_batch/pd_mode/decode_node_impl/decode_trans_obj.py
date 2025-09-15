import time
import psutil
import threading
from typing import List
from dataclasses import dataclass
from lightllm.utils.log_utils import init_logger
from ..task_queue import TaskQueue
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask, UpKVStatus, PDTransJoinInfo, PDTransLeaveInfo, KVMoveTaskGroup
from lightllm.utils.device_utils import kv_trans_use_p2p
from .decode_kv_move_manager import DecodeKVMoveManager
from lightllm.utils.time_utils import TimeChecker
from ..utils import join_if_alive, clear_queue

logger = init_logger(__name__)

KV_MOVE_MAX_NUM = 16


@dataclass
class KVTransConnectObj:
    connect_id: str = None
    prefill_node_id: int = None
    kv_trans_process: "KVTransProcess" = None
    pd_prefill_nccl_ip: str = None
    pd_prefill_nccl_port: int = None
    device_index: int = None
    manager: "DecodeKVMoveManager" = None
    has_error: bool = False
    ready_to_move_queue: TaskQueue = None
    kv_move_thread: threading.Thread = None
    move_finished_queue: TaskQueue = None
    put_to_radix_thread: threading.Thread = None
    timer_checker: TimeChecker = None

    def create(
        self,
        connect_id: str,
        prefill_node_id: str,
        pd_prefill_nccl_ip: str,
        pd_prefill_nccl_port: int,
        manager: "DecodeKVMoveManager",
    ):
        self.connect_id = connect_id
        self.device_index = manager.get_next_device_index()
        self.kv_trans_process = manager.kv_trans_processes[self.device_index]
        decode_node_id = manager.args.pd_node_id
        self.prefill_node_id = prefill_node_id
        self.decode_node_id = decode_node_id
        self.pd_prefill_nccl_ip = pd_prefill_nccl_ip
        self.pd_prefill_nccl_port = pd_prefill_nccl_port

        self.manager = manager
        self.timer_checker = TimeChecker(6)

        with self.kv_trans_process.device_lock:
            clear_queue(self.kv_trans_process.task_out_queue)
            self.kv_trans_process.task_in_queue.put(
                PDTransJoinInfo(
                    prefill_id=prefill_node_id,
                    prefill_device_id=-1,
                    pd_prefill_nccl_ip=pd_prefill_nccl_ip,
                    pd_prefill_nccl_port=pd_prefill_nccl_port,
                    decode_id=decode_node_id,
                    decode_device_id=self.device_index,
                    connect_id=self.connect_id,
                )
            )
            assert self.kv_trans_process.task_out_queue.get(timeout=60) == "nccl_ok"

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

    # ==================================================================================
    # 处理接受所有进行 kv 传输的请求，完成后，将请求放入到 move_finished_queue 中
    # ==================================================================================

    def _transfer_kv(self, move_tasks: List[KVMoveTask]):
        with self.kv_trans_process.device_lock:
            clear_queue(self.kv_trans_process.task_out_queue)
            kv_move_group = KVMoveTaskGroup(tasks=move_tasks.copy(), connect_id=self.connect_id)
            kv_move_group.connect_id = self.connect_id
            self.kv_trans_process.task_in_queue.put(kv_move_group, timeout=10)
            assert self.kv_trans_process.task_out_queue.get(timeout=60) == "ok"
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

            move_tasks: List[KVMoveTask] = move_tasks[0]
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

            finally:
                self.manager.put_to_fail_release_task_queue(move_tasks)

        logger.error(f"{func_name}  thread quit")
        return

    # ==================================================================================
    # 将传输完成的请求，放入到 radix cache 中进行管理。
    # ==================================================================================

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
                self.timer_to_check_status(raise_exception=True)
                # random to check stats
                self.manager._put_kv_received_to_radix_cache(move_tasks.copy())
                for task in move_tasks.copy():
                    logger.info(
                        f"{func_name} put kv to radix cache ok, req_id: {task.id()} cost_time {task.get_cost_time()} s"
                    )
                    self.manager.up_status_in_queue.put(
                        UpKVStatus(
                            group_request_id=task.group_request_id,
                            dp_index=task.decode_dp_index,
                            pd_master_node_id=task.pd_master_node_id,
                        )
                    )
                    logger.info(f"{func_name} up kv status req_id: {task.id()} finished")
                move_tasks.clear()

            except BaseException as e:
                logger.exception(str(e))
                self.set_has_error()
                self.move_finished_queue.clear_tasks()

            finally:
                self.manager.put_to_fail_release_task_queue(move_tasks)

        logger.error(f"{func_name} thread quit, info: {self.to_log_info()}")
        return

    # ==================================================================================
    # 错误处理检测操作的一些通用函数
    # ==================================================================================

    def timer_to_check_status(self, raise_exception=True):
        if self.timer_checker.has_exceeded():
            try:
                assert self.kv_trans_process.is_trans_process_health()
            except BaseException as e:
                logger.error(f"pid {self.kv_trans_process.process.pid} check failed")
                logger.exception(str(e))

                self.set_has_error()
                if raise_exception:
                    raise e
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

        if self.ready_to_move_queue is not None:
            self.ready_to_move_queue.has_error = True

        if self.move_finished_queue is not None:
            self.move_finished_queue.has_error = True

        if self.manager is not None:
            self.manager.remove_trans_obj(self.connect_id)
        return

    def __del__(self):
        logger.error(f"trans obj del start, info: {self.to_log_info()}")

        try:
            self.set_has_error()

            join_if_alive(self.kv_move_thread)
            join_if_alive(self.put_to_radix_thread)

            if self.connect_id is not None and self.kv_trans_process is not None:
                self.kv_trans_process.task_in_queue.put(
                    PDTransLeaveInfo(
                        decode_id=self.decode_node_id, prefill_id=self.prefill_node_id, connect_id=self.connect_id
                    )
                )

            if self.ready_to_move_queue is not None:
                self.ready_to_move_queue.clear_tasks()
            if self.move_finished_queue is not None:
                self.move_finished_queue.clear_tasks()

        except BaseException as e:
            logger.exception(str(e))

        logger.error(f"trans obj deled, info: {self.to_log_info()}")

    def to_log_info(self):
        log = f"connect_id: {self.connect_id} "
        log += f"decode_node_id: {self.decode_node_id} "
        log += f"prefill_node_id: {self.prefill_node_id} "
        log += f"device_index: {self.device_index} "
        return log


@dataclass
class KVTransProcess:
    process: mp.Process = None
    # 需要每个卡有一个锁来规划每次只能有一个 connection obj 操作对应显卡上的传输任务。
    device_lock: threading.Lock = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    device_id: int = None

    def init_all(self, device_id: int, manager: "DecodeKVMoveManager"):
        self.device_lock = threading.Lock()
        self.device_id = device_id
        self.task_in_queue = mp.Queue()
        self.task_out_queue = mp.Queue()

        try:
            from .decode_trans_process import start_decode_trans_process

            self.process = start_decode_trans_process(
                manager.args,
                device_id,
                self.task_in_queue,
                self.task_out_queue,
                manager.mem_queues,
            )
            assert self.task_out_queue.get(timeout=30) == "proc_start"
            manager._put_mem_manager_to_mem_queue()
            assert self.task_out_queue.get(timeout=60) == "get_mem_managers_ok"

            return True

        except Exception as e:
            logger.warning(f"Failed start kv trans process for device {device_id}: {e}")
            logger.exception(str(e))
            return False

    def is_trans_process_health(self):
        try:
            process = psutil.Process(self.process.pid)
            if not (process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
                logger.error(f"kv trans process for device: {self.device_id} dead!!!")
                return False
            else:
                return True
        except:
            return False

    def killself(self):
        self.process.kill()

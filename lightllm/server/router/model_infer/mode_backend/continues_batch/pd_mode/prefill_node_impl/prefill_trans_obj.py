import time
import rpyc
import copy
import uuid
import numpy as np
import psutil
import threading
from dataclasses import dataclass
from typing import List, Dict, Union
from lightllm.utils.log_utils import init_logger
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask, PDTransJoinInfo, PDTransLeaveInfo, KVMoveTaskGroup
from rpyc.utils.classic import obtain
from ..task_queue import TaskQueue
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.time_utils import TimeChecker
from .prefill_kv_move_manager import PrefillKVMoveManager
from lightllm.utils.net_utils import find_available_port
from ..utils import join_if_alive, clear_queue

logger = init_logger(__name__)


@dataclass
class KVTransConnectObj:
    connect_id: str = None
    decode_node_id: int = None
    rpyc_conn: object = None  # rpyc_con 的连接对象
    kv_trans_process: "KVTransProcess" = None
    device_index: int = None  # 使用的gpu序号
    manager: "PrefillKVMoveManager" = None
    has_error: bool = False
    request_kv_trans_task_queue: TaskQueue = None
    request_thread: threading.Thread = None
    ready_kv_trans_task_queue: TaskQueue = None
    kv_trans_thread: threading.Thread = None
    timer_checker: TimeChecker = None

    # ==================================================================================
    # 构建传输通信对象
    # ==================================================================================

    def create(
        self, decode_node_id: int, decode_node_ip: str, decode_node_rpyc_port: int, manager: "PrefillKVMoveManager"
    ):
        device_index = manager.get_next_device_index()  # 分配使用的显卡index
        self.kv_trans_process = manager.kv_trans_processes[device_index]
        prefill_node_id = manager.args.pd_node_id
        self.connect_id = str(uuid.uuid4())
        self.decode_node_id = decode_node_id
        self.prefill_node_id = prefill_node_id
        self.device_index = device_index
        self.manager = manager
        self.timer_checker = TimeChecker(6)

        con = rpyc.connect(
            host=decode_node_ip,
            port=decode_node_rpyc_port,
            config={"allow_pickle": True, "sync_request_timeout": 60},
            keepalive=True,
        )

        self.rpyc_conn = con

        # 创建 nccl 连接
        with self.kv_trans_process.device_lock:
            clear_queue(self.kv_trans_process.task_out_queue)

            self.kv_trans_process.task_in_queue.put(
                PDTransJoinInfo(
                    prefill_id=prefill_node_id,
                    prefill_device_id=device_index,
                    pd_prefill_nccl_ip=manager.host_ip,
                    pd_prefill_nccl_port=self.kv_trans_process.kv_trans_port,
                    decode_id=decode_node_id,
                    decode_device_id=-1,
                    connect_id=self.connect_id,
                )
            )

            # 异步调用, 让decode节点建立与prefill节点进行nccl通信的进程
            max_kv_trans_token_num = obtain(
                con.root.build_trans_connect(
                    prefill_node_id,
                    manager.host_ip,
                    self.kv_trans_process.kv_trans_port,
                    manager.args.max_total_token_num,
                    self.connect_id,
                )
            )
            self.max_kv_trans_token_num = max_kv_trans_token_num
            assert self.kv_trans_process.task_out_queue.get(timeout=60) == "nccl_ok"

        self.request_kv_trans_task_queue = TaskQueue(
            get_func=self._get_request_tasks, fail_func=self.manager.put_to_release_task_queue
        )
        self.request_thread = threading.Thread(target=self.request_kv_trans_loop, daemon=True)
        self.request_thread.start()

        self.ready_kv_trans_task_queue = TaskQueue(lambda datas: datas[0:1], self.manager.put_to_release_task_queue)
        self.kv_trans_thread = threading.Thread(target=self.kv_trans_handle_loop, daemon=True)
        self.kv_trans_thread.start()

        logger.info(f"create KVTransConnectObj success: {self.to_log_info()}")
        return

    def _get_request_tasks(self, datas: List[KVMoveTask]):
        """
        根据可以p和d节点间协商得到的 max_kv_trans_token_num 限制，将排队等待
        传输的请求打包成一个可以传输的list组。
        """
        ans_list = []
        token_num = 0
        for task in datas:
            if token_num + len(task.prefill_token_indexes) <= self.max_kv_trans_token_num:
                ans_list.append(task)
                token_num += len(task.prefill_token_indexes)
            else:
                break
        return ans_list

    # ==================================================================================
    # 与 decode 节点进行元数据交互，申请锁定资源准备进行kv的传输
    # ==================================================================================
    def request_kv_trans_loop(self):
        func_name = self.request_kv_trans_loop.__name__

        while not self.has_error:
            move_tasks: List[KVMoveTask] = self.request_kv_trans_task_queue.get_tasks(
                log_tag="request_kv_trans_task_queue"
            )
            if len(move_tasks) == 0:
                self.timer_check_status(raise_exception=False)
                time.sleep(0.01)
                continue
            try:
                self.timer_check_status(raise_exception=True)
                for move_task in move_tasks:
                    move_task.connect_id = self.connect_id
                    logger.info(
                        f"{func_name} get task {move_task.to_prefill_log_info()} "
                        f"queue time {move_task.get_cost_time()} s "
                    )

                trans_move_tasks = [copy.copy(move_task) for move_task in move_tasks]
                for trans_move_task in trans_move_tasks:
                    trans_move_task.prefill_token_indexes = None

                mark_start = time.time()
                move_kv_lens = self.rpyc_conn.root.request_data_transfer(trans_move_tasks)
                move_kv_lens = obtain(move_kv_lens)
                request_data_transfer_cost_time = time.time() - mark_start

                logger.info(
                    f"{func_name} request_data_transfer ok, {move_tasks[0].to_prefill_log_info()}"
                    f" cost time: {request_data_transfer_cost_time} s"
                )

                ok_trans_list = []
                for i, move_task in enumerate(move_tasks.copy()):
                    if move_kv_lens[i] is not None:
                        move_task.move_kv_len = move_kv_lens[i]
                        ok_trans_list.append(move_task)
                        move_tasks.remove(move_task)
                    else:
                        logger.info(f"prefill node kv move task req_id: {move_task.id()} not send, decode is busy")

                if ok_trans_list:
                    self.ready_kv_trans_task_queue.put(
                        ok_trans_list, error_handle_func=self.manager.put_to_release_task_queue
                    )

            except BaseException as e:
                logger.exception(str(e))
                self.set_has_error()
                self.request_kv_trans_task_queue.clear_tasks()

            finally:
                # 将没有申请成功的请求放入到释放队列中
                self.manager.put_to_release_task_queue(move_tasks)

        logger.error(f"{func_name}, {self.to_log_info()} thread quit")
        return

    # ==================================================================================
    # 将准备好 kv 传输的请求进行 kv 传输
    # ==================================================================================
    def _transfer_kv(self, move_tasks: List[KVMoveTask]):
        with self.kv_trans_process.device_lock:
            clear_queue(self.kv_trans_process.task_out_queue)
            kv_move_group = KVMoveTaskGroup(tasks=move_tasks.copy(), connect_id=self.connect_id)
            self.kv_trans_process.task_in_queue.put(kv_move_group, timeout=10)
            assert self.kv_trans_process.task_out_queue.get(timeout=60) == "ok"
            self.manager.put_to_release_task_queue(move_tasks)

            logger.info(
                f"_transfer_kv data ok, req_id: {move_tasks[0].id()}"
                f" cost total time: {move_tasks[0].get_cost_time()} s"
            )
            move_tasks.clear()

    def kv_trans_handle_loop(self):
        func_name = self.kv_trans_handle_loop.__name__
        while not self.has_error:
            move_tasks: List[List[KVMoveTask]] = self.ready_kv_trans_task_queue.get_tasks(
                log_tag="ready_kv_trans_task_queue"
            )
            if len(move_tasks) == 0:
                self.timer_check_status(raise_exception=False)
                time.sleep(0.01)
                continue

            if len(move_tasks) != 1:
                logger.error(f"error get kv trans move_tasks, must be 1, get {len(move_tasks)}")
                assert len(move_tasks) == 1

            move_tasks: List[KVMoveTask] = move_tasks[0]

            try:
                self.timer_check_status(raise_exception=True)
                for move_task in move_tasks:
                    logger.info(
                        f"{func_name} get task {move_task.to_prefill_log_info()} to start kv move"
                        f"queue time {move_task.get_cost_time()} s "
                    )

                if not kv_trans_use_p2p():
                    with self.manager.kv_trans_lock:
                        self._transfer_kv(move_tasks)
                else:
                    self._transfer_kv(move_tasks)
            except BaseException as e:
                logger.exception(str(e))
                self.set_has_error()
                self.ready_kv_trans_task_queue.clear_tasks()
            finally:
                self.manager.put_to_release_task_queue(move_tasks)

        logger.error(f"trans kv thread, {self.to_log_info()} thread quit")
        return

    # ==================================================================================
    # 错误处理检测操作的一些通用函数
    # ==================================================================================

    def has_error_status(self):
        try:
            assert self.has_error is False
            assert self.request_thread.is_alive()
            assert self.kv_trans_thread.is_alive()
        except BaseException as e:
            logger.exception(str(e))
            self.set_has_error()
            return True

        return False

    def timer_check_status(self, raise_exception=True):
        if self.timer_checker.has_exceeded():
            try:
                self.rpyc_conn.root.check_alive()
                assert self.kv_trans_process.is_trans_process_health()
            except BaseException as e:
                logger.error(f"pid {self.kv_trans_process.process.pid} check failed")
                logger.exception(str(e))

                self.set_has_error()
                if raise_exception:
                    raise e

        return

    def set_has_error(self):
        """
        将当前传输对象标记为有错误，这样可以防止请求放入到处理队列中
        """
        self.has_error = True

        if self.request_kv_trans_task_queue is not None:
            self.request_kv_trans_task_queue.has_error = True

        if self.ready_kv_trans_task_queue is not None:
            self.ready_kv_trans_task_queue.has_error = True

        if self.manager is not None:
            self.manager.remove_trans_obj(self.connect_id)
        return

    def __del__(self):
        """
        函数中有很多判断是否是None的操作，主要是为了避免一些异常流程的del行为不报错。
        """
        logger.error(f"trans obj del start, info: {self.to_log_info()}")

        try:
            self.set_has_error()

            join_if_alive(self.request_thread)
            join_if_alive(self.kv_trans_thread)

            # 将未处理的请求，清理掉，clear_tasks 会将没处理完的请求
            # 放入到 manager 资源释放队列中
            if self.request_kv_trans_task_queue is not None:
                self.request_kv_trans_task_queue.clear_tasks()
            if self.ready_kv_trans_task_queue is not None:
                self.ready_kv_trans_task_queue.clear_tasks()

            # 传输进程清理掉 nccl 连接
            if self.connect_id is not None:
                self.kv_trans_process.task_in_queue.put(
                    PDTransLeaveInfo(
                        decode_id=self.decode_node_id, prefill_id=self.prefill_node_id, connect_id=self.connect_id
                    )
                )

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
    kv_trans_port: int = None

    def init_all(self, device_id: int, manager: "PrefillKVMoveManager"):
        self.device_id = device_id
        self.device_lock = threading.Lock()
        self.task_in_queue = mp.Queue()
        self.task_out_queue = mp.Queue()
        self.kv_trans_port = find_available_port(manager.args.pd_p_allowed_port_min, manager.args.pd_p_allowed_port_max)

        try:
            from .prefill_trans_process import start_prefill_trans_process

            self.process = start_prefill_trans_process(
                manager.args,
                manager.host_ip,
                self.kv_trans_port,
                device_id,
                self.task_in_queue,
                self.task_out_queue,
            )
            assert self.task_out_queue.get(timeout=30) == "proc_start"
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

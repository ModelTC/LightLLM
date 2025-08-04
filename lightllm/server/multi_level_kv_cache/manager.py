import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import inspect
import pickle
import time
import threading
import concurrent.futures
from typing import List
from lightllm.server.core.objs import ShmReqManager, Req, StartArgs
from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.utils.graceful_utils import graceful_registry
from .cpu_cache_client import CpuKvCacheClient
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MultiLevelKVCacheManager:
    def __init__(
        self,
        args,
        detokenization_port,
        router_port,
    ):
        self.args: StartArgs = args
        context = zmq.Context(2)
        self.recv_from_pre_module = context.socket(zmq.PULL)
        self.recv_from_pre_module.bind(f"{args.zmq_mode}127.0.0.1:{detokenization_port}")

        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.bind(f"{args.zmq_mode}127.0.0.1:{router_port}")
        logger.info(f"pub_to_httpserver sendhwm {self.send_to_router.getsockopt(zmq.SNDHWM)}")
        self.cpu_cache_client = CpuKvCacheClient(init_shm_data=True)
        self.shm_req_manager = ShmReqManager()
        # 控制同时进行cpu cache 匹配操作的数量。
        self.semaphore = threading.Semaphore(3)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
        # 控制 cpu cache time out的时间，如果超过这个时间无法获取信号量则直接转发。
        self.cpu_cache_time_out = 0.3
        # lock 用于控制对 recv_queue 和 transfer_queue 的访问。
        self.queue_lock = threading.Lock()
        self.recv_queue: List[GroupReqIndexes] = []
        self.transfer_queue: List[GroupReqIndexes] = []
        self.transfer_thread = threading.Thread(target=self.transfer_loop, daemon=True)
        self.transfer_thread.start()
        self.cpu_cache_thread = threading.Thread(target=self.cpu_cache_hanle_loop, daemon=True)
        self.cpu_cache_thread.start()
        return

    def cpu_cache_hanle_loop(self):
        while True:
            try:
                if len(self.recv_queue) == 0:
                    time.sleep(0.003)
                    continue

                with self.queue_lock:
                    current_group_req = self.recv_queue[0]
                    self.recv_queue = self.recv_queue[1:]

                self.executor.submit(self._handle_group_req_cpu_cache_match, current_group_req, time.time())
            except BaseException as e:
                logger.exception(str(e))
        return

    def _handle_group_req_cpu_cache_match(self, group_req_indexes: GroupReqIndexes, start_time: float):
        """
        match cpu cache pages
        """
        # 进行超时判定，如果太长时间拿不到信号量，则说明匹配任务繁忙，
        # 放弃进行 cpu cache page 的匹配。
        while True:
            current_time = time.time()
            if current_time - start_time >= self.cpu_cache_time_out:
                with self.queue_lock:
                    self.transfer_queue.append(group_req_indexes)
                return

            if self.semaphore.acquire(blocking=False):
                break
            else:
                time.sleep(0.005)

        reqs_shm_index = group_req_indexes.shm_req_indexes
        reqs = [self.shm_req_manager.get_req_obj_by_index(index) for index in reqs_shm_index]
        req: Req = reqs[0]

        # 对每个请求进行cpu cache page 的匹配操作。
        for req in reqs:
            # diverse_mode 只有主请求一个初始化 cpu cache 信息。
            if self.args.diverse_mode and req.request_id != req.group_req_id:
                continue

            self.cpu_cache_client.lock.acquire_sleep1ms()
            req: Req = req
            finded_page_indexes = []
            for token_chuncked_hash_value in req.token_hash_list.get_all():
                page_index, ready = self.cpu_cache_client.query_one_page(token_chuncked_hash_value)
                if page_index is not None:
                    assert ready
                    finded_page_indexes.append(page_index)
                else:
                    break
            self.cpu_cache_client.lock.release()

            # 等待所有的cpu cache 页面ready
            while not self.cpu_cache_client.check_allpages_ready(finded_page_indexes):
                time.sleep(0.01)

            req.cpu_cache_match_page_indexes.fill(finded_page_indexes)

        for req in reqs:
            self.shm_req_manager.put_back_req_obj(req)

        # 释放信号量
        self.semaphore.release()

        # 将请求放入转发队列
        with self.queue_lock:
            self.transfer_queue.append(group_req_indexes)
        return

    def transfer_loop(self):
        while True:
            try:
                if len(self.transfer_queue) != 0:
                    with self.queue_lock:
                        for e in self.transfer_queue:
                            self.send_to_router.send_pyobj(e, protocol=pickle.HIGHEST_PROTOCOL)
                        self.transfer_queue.clear()
                else:
                    time.sleep(0.005)
            except BaseException as e:
                logger.exception(str(e))
        return

    def recv_loop(self):
        try:
            recv_max_count = 128

            while True:
                recv_objs = []
                try:
                    # 一次最多从 zmq 中取 recv_max_count 个请求，防止 zmq 队列中请求数量过多导致阻塞了主循环。
                    for _ in range(recv_max_count):
                        recv_obj: GroupReqIndexes = self.recv_from_pre_module.recv_pyobj(zmq.NOBLOCK)
                        assert isinstance(recv_obj, GroupReqIndexes)
                        recv_objs.append(recv_obj)

                    # 当队列中存在较多的请求时，将一次接受的数量上调
                    recv_max_count = min(int(recv_max_count * 1.3), 256)
                except zmq.ZMQError:
                    # 当队列已经开始清空的时候，将一次接受的数量下调
                    recv_max_count = 128

                with self.queue_lock:
                    self.recv_queue.extend(recv_objs)

                time.sleep(0.003)

        except Exception as e:
            logger.exception(f"detoken process has exception {str(e)}")
        return


def start_detokenization_process(args, detokenization_port, router_port, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        manager = MultiLevelKVCacheManager(
            args=args,
            detokenization_port=detokenization_port,
            router_port=router_port,
        )
    except Exception as e:
        pipe_writer.send(str(e))
        raise

    pipe_writer.send("init ok")
    manager.recv_loop()
    return

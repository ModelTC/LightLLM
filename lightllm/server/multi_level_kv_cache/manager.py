import uvloop
import asyncio

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import inspect
import pickle
import time
import threading
import concurrent.futures
from queue import Queue
from lightllm.server.core.objs import ShmReqManager, Req, StartArgs
from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.utils.graceful_utils import graceful_registry
from .cpu_cache_client import CpuKvCacheClient
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MultiLevelKVCacheManager:
    def __init__(
        self,
        args: StartArgs,
    ):
        self.args: StartArgs = args
        context = zmq.Context(2)
        self.zmq_recv_socket = context.socket(zmq.PULL)
        self.zmq_recv_socket.bind(f"{args.zmq_mode}127.0.0.1:{args.multi_level_kv_cache_port}")

        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"{args.zmq_mode}127.0.0.1:{args.router_port}")
        logger.info(f"send_to_router sendhwm {self.send_to_router.getsockopt(zmq.SNDHWM)}")
        self.cpu_cache_client = CpuKvCacheClient(only_create_meta_data=False, init_shm_data=True)
        self.shm_req_manager = ShmReqManager()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)
        # 控制进行 cpu cache 页面匹配的时间，超过时间则不再匹配，直接转发。
        self.cpu_cache_time_out = 0.5
        self.recv_queue = Queue(maxsize=1024)
        self.cpu_cache_thread = threading.Thread(target=self.cpu_cache_hanle_loop, daemon=True)
        self.cpu_cache_thread.start()
        return

    def cpu_cache_hanle_loop(self):
        while True:
            try:
                current_group_req = self.recv_queue.get()

                self.executor.submit(self._handle_group_req_cpu_cache_match, current_group_req, time.time())
            except BaseException as e:
                logger.exception(str(e))
        return

    def _handle_group_req_cpu_cache_match(self, group_req_indexes: GroupReqIndexes, start_time: float):
        """
        match cpu cache pages
        """
        # 超时时，放弃进行 cpu cache page 的匹配。
        current_time = time.time()
        if current_time - start_time >= self.cpu_cache_time_out:
            self.send_to_router.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
            logger.warning(
                f"cpu cache match time out {current_time - start_time}s, group_req_id: {group_req_indexes.group_req_id}"
            )
            return

        reqs_shm_index = group_req_indexes.shm_req_indexes
        reqs = [self.shm_req_manager.get_req_obj_by_index(index) for index in reqs_shm_index]

        # 对每个请求进行cpu cache page 的匹配操作。
        for req in reqs:
            # diverse_mode 只有主请求一个初始化 cpu cache 信息。
            if self.args.diverse_mode and req.request_id != req.group_req_id:
                continue

            if req.is_aborted:
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

            # 等待所有的 cpu cache 页面ready
            while not self.cpu_cache_client.check_allpages_ready(finded_page_indexes):
                time.sleep(0.01)

            req.cpu_cache_match_page_indexes.fill(finded_page_indexes)

        for req in reqs:
            self.shm_req_manager.put_back_req_obj(req)

        self.send_to_router.send_pyobj(group_req_indexes, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def recv_loop(self):
        try:
            recv_max_count = 128

            while True:
                recv_objs = []
                try:
                    # 一次最多从 zmq 中取 recv_max_count 个请求，防止 zmq 队列中请求数量过多导致阻塞了主循环。
                    for _ in range(recv_max_count):
                        recv_obj: GroupReqIndexes = self.zmq_recv_socket.recv_pyobj(zmq.NOBLOCK)
                        assert isinstance(recv_obj, GroupReqIndexes)
                        recv_objs.append(recv_obj)

                        start_time = recv_obj.time_mark
                        logger.info(
                            f"multi_level_kv_cache recive group req id {recv_obj.group_req_id} "
                            f"cost time {time.time() - start_time} s"
                        )

                    # 当队列中存在较多的请求时，将一次接受的数量上调
                    recv_max_count = min(int(recv_max_count * 1.3), 256)
                except zmq.ZMQError:
                    # 当队列已经开始清空的时候，将一次接受的数量下调
                    recv_max_count = 128

                for recv_obj in recv_objs:
                    self.recv_queue.put(recv_obj)

                if len(recv_objs) == 0:
                    time.sleep(0.01)

        except Exception as e:
            logger.exception(f"detoken process has exception {str(e)}")
        return


def start_multi_level_kv_cache_manager(args, pipe_writer):
    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        manager = MultiLevelKVCacheManager(
            args=args,
        )
    except Exception as e:
        pipe_writer.send(str(e))
        raise

    pipe_writer.send("init ok")
    manager.recv_loop()
    return

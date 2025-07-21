import time
import zmq
import zmq.asyncio
import inspect
import pickle
import torch.multiprocessing as mp
import threading
import asyncio
from typing import List
from dataclasses import dataclass
from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.utils.log_utils import init_logger, log_time_ready
from lightllm.utils.graceful_utils import graceful_registry
from .disk_cache_server import DiskCacheClient
from lightllm.server.core.objs import ShmReqManager
from .io_objs import ShmReqInfo, GroupReqInfo
from lightllm.server.core.objs import Req

logger = init_logger(__name__)

class HiRadixCacheManagerServer:
    def __init__(
            self, args, mem_queues: List[mp.Queue], radix_locks: List[mp.Lock], router_port: int):
        self.args = args
        self.mem_queues = mem_queues
        self.radix_locks = radix_locks
        self.node_world_size = args.tp // args.nnodes
        self.disk_cache_processes = []
        self.ports = args.hiradix_cache_ports
        self.cache_server_client = []
        context = zmq.asyncio.Context(3)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        recv_from_http_port, recv_from_router_port = self.args.hiradix_server_ports
        self.recv_from_httpserver.bind(f"{args.zmq_mode}127.0.0.1:{recv_from_http_port}")
        self.clients: List[DiskCacheClient] = []
        self.send_to_router = context.socket(zmq.PUSH)
        self.send_to_router.connect(f"{args.zmq_mode}127.0.0.1:{router_port}")
        self.recv_from_router = context.socket(zmq.PULL)
        self.recv_from_router.bind(f"{args.zmq_mode}127.0.0.1:{recv_from_router_port}")
        self.shm_req_manager = ShmReqManager()

    
    async def asyn_init(self):
        self.pull_queue = asyncio.Queue()
        self.push_queue = asyncio.Queue()
        
    async def start_all(self):
        from lightllm.server.router.dynamic_prompt.hiradix.disk_cache_server import start_disk_cache_server_process
        for rank_in_node in range(self.node_world_size):
            client = await start_disk_cache_server_process(
                self.args,
                device_id=rank_in_node,
                node_word_size=self.node_world_size,
                mem_queue=self.mem_queues[rank_in_node],
                radix_lock=self.radix_locks[rank_in_node],
                port=self.ports[rank_in_node]
            )
            self.clients.append(client)
    
    async def pull_cache(self, group_req):
        tasks = []
        group_req_info = GroupReqInfo(
            group_req_id=group_req.group_req_id,
            shm_req_indexes=group_req.shm_req_indexes
        ).to_dict()
        for client in self.clients:
            task = client.pull(group_req_info)
            tasks.append(task)
        all_results = await asyncio.gather(*tasks)
        logger.info(f"pull cache results {all_results}")
        await self.send_to_router.send_pyobj(group_req, protocol=pickle.HIGHEST_PROTOCOL)

    async def push_cache(self, req_info):
        tasks = []
        for client in self.clients:
            task = client.push(req_info)
            tasks.append(task)
        all_results = await asyncio.gather(*tasks)
        req: Req = self.shm_req_manager.get_req_obj_by_index(req_info["shm_req_index"])
        assert req.radix_status.is_write_done()
        req.radix_status.set_finished()
        logger.info(f"push cache results {all_results}")

    async def pull_woker(self):
        while True:
            req: GroupReqInfo = await self.pull_queue.get()
            await self.pull_cache(req)
            await asyncio.sleep(0.01)

    async def push_woker(self):
        while True:
            req: ShmReqInfo = await self.push_queue.get()
            await self.push_cache(req.to_dict())
            await asyncio.sleep(0.01)

    async def run(self):
        await self.asyn_init()
        await asyncio.gather(
            self.loop_for_netio_req_to_pull(),
            self.pull_woker(),
            self.loop_for_netio_req_to_push(),
            self.push_woker()
        )

    async def loop_for_netio_req_to_push(self):
        while True:
            recv_req: ShmReqInfo = await self.recv_from_router.recv_pyobj()
            if isinstance(recv_req, ShmReqInfo):
                await self.push_queue.put(recv_req)
            else:
                raise ValueError(f"Invalid request: {recv_req}")

    async def loop_for_netio_req_to_pull(self):
        while True:
            recv_req: GroupReqIndexes = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, GroupReqIndexes):
                await self.pull_queue.put(recv_req)
            else:
                raise ValueError(f"Invalid request: {recv_req}")

def _init_env_server(
    args,
    mem_queues,
    radix_locks: List[mp.Lock],
    init_event: mp.Event,
    router_port: int
):
    graceful_registry(inspect.currentframe().f_code.co_name)
    hiradix_cache_manager = HiRadixCacheManagerServer(
        args, 
        mem_queues=mem_queues, 
        radix_locks=radix_locks, 
        router_port=router_port
    )
    asyncio.run(hiradix_cache_manager.start_all())
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        init_event.set()
        loop.run_until_complete(hiradix_cache_manager.run())
    except Exception as e:
        logger.error(f"hiradix server error happend {e}")
    return

def start_hiradix_cache_manager_process_server(
    args,
    radix_mem_queues: List[mp.Queue],
    radix_locks: List[mp.Lock],
    router_port: int
):
    """
    Start the HiRadix cache manager process.
    """
    init_event = mp.Event()
    proc = mp.Process(target=_init_env_server, args=(args, radix_mem_queues, radix_locks, init_event, router_port))
    proc.start()
    init_event.wait()
    logger.info(f"HiRadix cache manager process started")
    assert proc.is_alive()
    return
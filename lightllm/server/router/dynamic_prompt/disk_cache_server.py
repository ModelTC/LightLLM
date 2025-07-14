import torch
import time
import tempfile
import rpyc
import zmq
import inspect
import asyncio
import threading
import numpy as np
import torch.multiprocessing as mp
from typing import List, Union
from rpyc.utils.server import ThreadedServer
from os.path import join
from typing import Tuple, Dict, Set, List
from lightllm.utils.log_utils import init_logger
from enum import Enum
from .shared_arr import SharedArray
from .io_objs import ShmReqInfo, GroupReqInfo
from lightllm.server.core.objs.io_objs import GroupReqIndexes
from lightllm.server.core.objs import ShmReqManager
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.common.radixmem_buffer import RadixMemoryBuffer
from lightllm.server.core.objs import Req, RadixStatus

logger = init_logger(__name__)

def wait_until_ready(task, timeout=10.0, check_interval=0.01):
    start_time = time.time()
    while not task.ready():
        time.sleep(check_interval)
        if time.time() - start_time > timeout:
            logger.error("Current kv cache task not ready in time")
            return False
    return True

class RemoteCacheManager:
    def __init__(self, unique_name: str, rank_in_node: int, mem_manager):
        tmp_dir = tempfile.mkdtemp(prefix=f"cache_{unique_name}_{rank_in_node}")
        self.cache_file = join(tmp_dir, "cache_file")
        all_buffers = mem_manager.kv_buffer
        all_buffers = all_buffers.view(all_buffers.shape[0], all_buffers.shape[1], -1)
        from kvcache.python.jit import PyLocalCacheService

        self.py_cache_service = PyLocalCacheService(
            file=self.cache_file,
            storage_size=128 * (1024 ** 3),  # 128GB
            num_shard=32,
            kvcache_tensor=all_buffers,
            num_worker=8
        )

    def insert(self, tokens, kv_page_indexer, start_pos=0):
        t = self.py_cache_service.create(
                tokens=tokens, 
                kv_page_indexer=kv_page_indexer, 
                mode="w",
                start_pos=start_pos)
        res = wait_until_ready(t)
        if not res:
            self.py_cache_service.az5(t)

    def read(self, tokens, kv_page_indexer, start_pos=0):
        t = self.py_cache_service.create(
                tokens=tokens, 
                kv_page_indexer=kv_page_indexer, 
                mode="r",
                start_pos=start_pos)
        res = wait_until_ready(t)
        return res

    def query(self, tokens):
        query_result = self.py_cache_service.query(tokens)
        max_len = 0
        for result in query_result:
            if result:
                max_len += 1
            else:
                break
        return max_len * self.block_size

    @property
    def block_size(self,):
        return self.py_cache_service.tokens_per_block


class DiskCacheService(rpyc.Service):
    def __init__(self, mem_manager=None, remote_cache_manager=None, shm_req_manager=None, rank_in_node=None):
        super().__init__()
        self.mem_manager = mem_manager
        self.remote_cache_manager = remote_cache_manager
        self.shm_req_manager = shm_req_manager
        self.rank_in_node = rank_in_node

    def exposed_push(self, req_info):
        req_info: ShmReqInfo = ShmReqInfo.from_dict(req_info)
        req: Req = self.shm_req_manager.get_req_obj_by_index(req_info.shm_req_index)
        req.link_prompt_ids_shm_array()
        assert req.radix_status.is_write_ready(self.rank_in_node), "radix cache is not ready" 
        input_token_ids = req.shm_prompt_ids.arr[0 : req.shm_cur_kv_len]
        keys = torch.tensor(input_token_ids, dtype=torch.int64, device="cpu")
        values = self.mem_manager.get_req_mem_index(req_info.request_id)
        index = torch.tensor(values, device="cpu", dtype=torch.int32)
        logger.info(f"_push_task_loop receive task keys {len(keys)} values {len(values)}")
        self.remote_cache_manager.insert(keys, index)
        self.mem_manager.free_req_index(req.request_id)
        self.set_reqs_radix_status([req], RadixStatus.NOT_READY)
        self.shm_req_manager.put_back_req_obj(req)
        return {"status": "ok"}

    def set_reqs_radix_status(self, reqs: List[Req], status: int):
        for req in reqs:
            req.radix_status.set_status(self.rank_in_node, status)
            logger.info(f"-->pull loop rank_in_node={self.rank_in_node} set req {req.group_req_id, req.request_id} radix status {req.radix_status.get_status(self.rank_in_node)}")

    def put_back_req_objs(self, reqs: List[Req]):
        for req in reqs:
            self.shm_req_manager.put_back_req_obj(req)

    def exposed_pull(self, group_req):
        group_req: GroupReqInfo = GroupReqInfo.from_dict(group_req)
        reqs: List[Req] = []
        for shm_req_index in group_req.shm_req_indexes:
            req: Req = self.shm_req_manager.get_req_obj_by_index(shm_req_index)
            reqs.append(req)
        req = reqs[0]
        req.link_prompt_ids_shm_array()
        keys = req.get_prompt_ids()
        query_len = self.remote_cache_manager.query(tokens=keys)
        if query_len == 0:
            self.set_reqs_radix_status(reqs, RadixStatus.NOCACHE)
            return {"query_len": 0, "kv_indices": []}
        index = self.mem_manager.alloc(query_len)
        self.remote_cache_manager.read(tokens=keys[:query_len], kv_page_indexer=index)
        self.mem_manager.set_req_mem_index(
            group_req.group_req_id, index.tolist()
        )
        self.set_reqs_radix_status(reqs, RadixStatus.READ_READY)
        self.put_back_req_objs(reqs)
        return {"query_len": query_len, "kv_indices": index.tolist()}


class DiskCacheClient:
    def __init__(self, rank_in_node: int, service=None, use_rpc=True, proc=None):
        self.rank_in_node = rank_in_node
        self.use_rpc = use_rpc
        self.service = service
        self.proc=proc
        if self.use_rpc:
            self._push = self._async_wraper(self.service.push)
            self._pull = self._async_wraper(self.service.pull)
        else:
            self._push = self.service.exposed_push
            self._pull = self.service.exposed_pull

    def _async_wraper(self, func):
        async_func = rpyc.async_(func)

        async def _wrapped(*args, **kwargs):
            result = async_func(*args, **kwargs)
            await asyncio.to_thread(result.wait)
            return result.value

        return _wrapped

    async def push(self, req_info: ShmReqInfo):
        if self.use_rpc:
            return await self._insert(req_info)
        else:
            return self._insert(req_info)

    async def pull(self, group_req: GroupReqInfo):
        if self.use_rpc:
            return await self._pull(group_req)
        else:
            return self._pull(group_req)


def start_cache_server(mem_manager, remote_cache_manager, shm_req_manager, rank_in_node, port, init_event):
    class CustomService(DiskCacheService):
        def __init__(self):
            super().__init__(mem_manager, remote_cache_manager, shm_req_manager, rank_in_node)

    def start():
        try:
            server = ThreadedServer(CustomService(), 
                                    port=port, 
                                    protocol_config={"allow_public_attrs": True, "allow_pickle": True})
            init_event.set()
            server.start()
        except Exception as e:
            logger.error(f"Failed to start ThreadedServer: {e}")

    t = threading.Thread(target=start, daemon=True)
    t.start()

    logger.info(f"DiskCacheService started on port {port}")
    return t


def _init_server(
    device_id,
    mem_queue,
    radix_lock: List[mp.Lock],
    init_event: mp.Event,
    port:int=18861
):
    from lightllm.utils.envs_utils import get_unique_server_name
    graceful_registry(inspect.currentframe().f_code.co_name)
    torch.cuda.set_device(device_id)
    mem_proties, shared_mem_data = mem_queue.get()
    mem_manager = RadixMemoryBuffer(
        mem_propties=mem_proties,
        shared_data=shared_mem_data,
        lock=radix_lock,
        rank_in_node=device_id
    )
    remote_cache_manager = RemoteCacheManager(
        unique_name=get_unique_server_name(),
        rank_in_node=device_id,
        mem_manager=mem_manager,
    )
    shm_req_manager = ShmReqManager()

    t = start_cache_server(
        mem_manager=mem_manager,
        remote_cache_manager=remote_cache_manager,
        shm_req_manager=shm_req_manager,
        rank_in_node=device_id,
        port=port,
        init_event=init_event
    )
    t.join() 
    return
    
async def start_disk_cache_server_process(
    args,
    device_id,
    node_word_size,
    mem_queue,
    radix_lock,
    port
):
    """
    Start the DiskCacheManager in process.
    """
    from lightllm.utils.envs_utils import get_unique_server_name
    if node_word_size == 1:
        mem_proties, shared_mem_data = mem_queue.get()
        mem_manager = RadixMemoryBuffer(
            mem_propties=mem_proties,
            shared_data=shared_mem_data,
            lock=radix_lock,
            rank_in_node=device_id
        )
        remote_cache_manager = RemoteCacheManager(
            unique_name=get_unique_server_name(),
            rank_in_node=device_id,
            mem_manager=mem_manager,
        )
        shm_req_manager = ShmReqManager()
        service = DiskCacheService(mem_manager, remote_cache_manager, shm_req_manager)
        client = DiskCacheClient(
            service=service,
            rank_in_node=0, 
            use_rpc=False
        )
        return client

    init_event = mp.Event()
    proc = mp.Process(target=_init_server, args=(device_id, mem_queue, radix_lock, init_event, port))
    proc.start()

    init_event.wait(timeout=60)

    max_wait_times = 20
    for i in range(max_wait_times):
        try:
            conn = rpyc.connect("localhost", port, config={"allow_pickle": True})
            break
        except Exception as e:
            asyncio.sleep(2)

    service = conn.root
    client = DiskCacheClient(
        rank_in_node=device_id,
        service=service,
        use_rpc=True,
        proc=proc
    )
    assert proc.is_alive()
    logger.info(f"disk cache process for device {device_id} start!")
    return client
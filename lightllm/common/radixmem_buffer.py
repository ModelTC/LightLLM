
import torch
from dataclasses import dataclass
import torch.multiprocessing as mp
from lightllm.utils.log_utils import init_logger
from typing import List, Union
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from multiprocessing.managers import DictProxy, ListProxy
from multiprocessing import Manager


logger = init_logger(__name__)

@dataclass
class SharedRadixMemoryData:
    kv_buffer: torch.Tensor
    mem_state: torch.Tensor
    req_mem_index: DictProxy
    lru_queue: ListProxy

@dataclass
class MemPropties:
    size: int
    dtype: torch.dtype
    head_num: int
    head_dim: int
    layer_num: int

shared_mem_data: SharedRadixMemoryData = None


def init_shared_data(mem_propties: MemPropties, device="cuda"):
    size, dtype, head_num, head_dim, layer_num = mem_propties.size, mem_propties.dtype, \
        mem_propties.head_num, mem_propties.head_dim, mem_propties.layer_num
    global shared_mem_data

    if device == "cuda":
        kv_buffer = torch.empty(
            (layer_num, size, head_num, head_dim),
            dtype=dtype,
            device="cuda"
        )
    else:
        kv_buffer = torch.empty(
            (layer_num, size, head_num, head_dim),
            dtype=dtype,
            device="cpu"
        ).share_memory_()

    mem_state = torch.arange(size, dtype=torch.int32).share_memory_()
    manager = Manager()
    req_mem_index = manager.dict()
    lru_queue = manager.list()
    
    shared_mem_data = SharedRadixMemoryData(
        kv_buffer=kv_buffer,
        mem_state=mem_state,
        req_mem_index=req_mem_index,
        lru_queue=lru_queue
    )

def get_shared_data() -> SharedRadixMemoryData:
    """Get the shared memory data."""
    global shared_mem_data
    if shared_mem_data is None:
        raise RuntimeError("Shared memory data has not been initialized. Call init_shared_data first.")
    return shared_mem_data

class RadixMemoryBuffer:
    def __init__(self, mem_propties: MemPropties, shared_data: SharedRadixMemoryData = None, lock: mp.Lock = None, device="cuda",
                 rank_in_node=None):
        size, dtype, head_num, head_dim, layer_num = mem_propties.size, mem_propties.dtype, \
            mem_propties.head_num, mem_propties.head_dim, mem_propties.layer_num

        self.kv_buffer = shared_data.kv_buffer
        self.mem_state = shared_data.mem_state
        self.req_mem_index = shared_data.req_mem_index
        self.lock = lock if lock is not None else mp.Lock()

        #TODO profile size
        self.size = size  # token slot 个数
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.dtype = dtype

        can_use_mem_size = self.size
        mark_start = 0
        mark_end = self.size
        rank_in_node = rank_in_node if rank_in_node is not None else get_current_rank_in_node() 
        self.can_use_mem_size = SharedInt(
            f"{get_unique_server_name()}_radix_mem_manger_can_use_token_num_{rank_in_node}"
        )
        self.can_use_mem_size.set_value(can_use_mem_size)
        self.mark_start = SharedInt(
            f"{get_unique_server_name()}_radix_mem_manger_mark_start_{rank_in_node}"
        )
        self.mark_start.set_value(mark_start)

        self.mark_end = SharedInt(
            f"{get_unique_server_name()}_radix_mem_manger_mark_end_{rank_in_node}"
        )
        self.mark_end.set_value(mark_end)
        logger.info(f"create {get_unique_server_name()}_radix_mem_manger_can_use_token_num_{rank_in_node}")

    def _free(self, free_index: Union[torch.Tensor, List[int]]):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """
        end = self.mark_start.get_value()
        start = end - len(free_index)
        assert start >= 0, f"error free state start: {end} free len {len(free_index)}"

        if isinstance(free_index, list):
            self.mem_state.numpy()[start:end] = free_index
        else:
            # 从 gpu 到 cpu 的拷贝操作是流内阻塞操作
            self.mem_state[start:end] = free_index

        self.mark_start.set_value(end - len(free_index))

        self.can_use_mem_size.set_value(self.can_use_mem_size.get_value() + len(free_index))

        if self.can_use_mem_size.get_value() == len(self.mem_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size.get_value()}")
        return
    
    def free_req_index(self, req_id: int):
        """Free the memory index for a specific request ID."""
        with self.lock:
            if req_id not in self.req_mem_index:
                logger.warning(f"Request ID {req_id} not found in memory index.")
                return
            index = self.req_mem_index[req_id]
            self._free(index)
            logger.info(f"Freed memory index for request {req_id} size {len(index)}, left size {self.can_use_mem_size.get_value()}")
            del self.req_mem_index[req_id]

    def alloc(self, need_size) -> torch.Tensor:
        with self.lock:
            if need_size > self.mark_end.get_value() - self.mark_start.get_value():
                logger.error(
                    f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size.get_value()}"
                )
                raise RuntimeError(f"Not enough memory to allocate {need_size} tokens.")

            start = self.mark_start.get_value()
            end = start + need_size
            ans = self.mem_state[start:end]
            self.mark_start.set_value(start + need_size)

            self.can_use_mem_size.set_value(self.can_use_mem_size.get_value() - need_size)
            return ans

    def set_req_mem_index(self, req_id: int, index: List[int]):
        """Set the memory index for a specific request ID."""
        with self.lock:
            if req_id in self.req_mem_index:
                logger.info(f"Request ID {req_id} already exists. Overwriting index {self.req_mem_index[req_id]} with {index}.")
            self.req_mem_index[req_id] = index
            logger.info(f"radix mem buffer insert req {req_id}, current disk work num {self._get_current_work_num()}")

    def get_req_mem_index(self, req_id: int) -> List[int]:
        """Get the memory index for a specific request ID."""
        with self.lock:
            if req_id not in self.req_mem_index:
                logger.warning(f"Request ID {req_id} not found. Returning empty list.")
                return []
            return self.req_mem_index[req_id]

    def get_kv_buffer(self, index) -> torch.Tensor:
        with self.lock:
            return self.kv_buffer[:, index, :, :]

    def _get_current_work_num(self) -> int:
        return len(self.req_mem_index)

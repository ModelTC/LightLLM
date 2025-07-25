import torch
import time
import xxhash
import numpy as np
from typing import List, Dict, Tuple, Optional
import torch.multiprocessing as mp
from collections import OrderedDict

from .radixmem_buffer import MemPropties, init_shared_data, get_shared_data
from .radixmem_buffer import SharedRadixMemoryData, RadixMemoryBuffer

from lightllm.utils.log_utils import init_logger
logger = init_logger(__name__)

class RadixBufferManager:

    def __init__(self,
                 radix_buffer: RadixMemoryBuffer = None,
                 radix_mem_data: SharedRadixMemoryData = None,
                 lock: Optional[mp.Lock] = None,
                 max_entries: int = 10000,
                 chunk_size: int = 64
                 ):
        self.chunk_size = chunk_size
        self.max_entries = max_entries
        self.radix_buffer = radix_buffer
        self.lru_queue = radix_mem_data.lru_queue
        
        self.lock = lock if lock is not None else mp.Lock()

    def _compute_hash(self, tokens: List[int]) -> List[Tuple[int, List[int]]]:
        chunks = []
        hsum = xxhash.xxh3_64()
        cumulative_tokens = []
        
        for i in range(0, len(tokens), self.chunk_size):
            chunk = tokens[i:i + self.chunk_size]
            cumulative_tokens.extend(chunk)
            
            chunk_np = np.array(chunk, dtype=np.uint32)
            hsum.update(chunk_np.tobytes())
            
            current_hash = hsum.intdigest()
            chunks.append((current_hash, cumulative_tokens.copy()))
            
        return chunks

    def write(self, tokens: List[int], values: torch.Tensor, start_pos: int=0) -> None:
        with self.lock:
            index = start_pos // self.chunk_size
            chunks = self._compute_hash(tokens)
            
            values = values[index * self.chunk_size:]
            chunks = chunks[index:]
            for i, (hash_val, _) in enumerate(chunks):
                if hash_val not in self.radix_buffer.req_mem_index:
                    self.radix_buffer.req_mem_index[hash_val] = values[i * self.chunk_size : (i + 1) * self.chunk_size]
                self._update_lru_state(hash_val)

    def _update_lru_state(self, hash_val: int):
        if hash_val in self.lru_queue:
            self.lru_queue.remove(hash_val)
        self.lru_queue.append(hash_val)
        
        while len(self.lru_queue) > self.max_entries:
            self.lru_queue.pop(0)

    def _free_space(self, required_size: int) -> bool:
        current_free = self.radix_buffer.can_use_mem_size.get_value()
        
        if current_free >= required_size:
            return True
            
        need_to_free = required_size - current_free
        freed_size = 0
        
        while freed_size < need_to_free and len(self.lru_queue) > 0:
            evict_size = self._evict_lru()
            freed_size += evict_size
        
        final_free = self.radix_buffer.can_use_mem_size.get_value()
        return final_free >= required_size
    
    def alloc(self, required_size: int) -> bool:
        with self.lock:
            self._free_space(required_size)
            ans = self.radix_buffer.alloc(required_size)
            return ans

    def _evict_lru(self):
        if not self.lru_queue:
            return
        oldest_hash = self.lru_queue[0]
        
        evict_size = 0
        if oldest_hash in self.radix_buffer.req_mem_index:
            indices = self.radix_buffer.req_mem_index[oldest_hash]
            evict_size += len(indices)
            self.radix_buffer._free(indices)
            del self.radix_buffer.req_mem_index[oldest_hash]
        
        self.lru_queue.pop(0)
        return evict_size

    def query_cache(self, tokens: List[int]) -> int:
        with self.lock:
            chunks = self._compute_hash(tokens)
            if not chunks:
                return 0, []

            max_hit = 0
            mem_index = []
            for hash_val, _ in chunks:
                if hash_val in self.radix_buffer.req_mem_index:
                    index_val = self.radix_buffer.req_mem_index[hash_val]
                    mem_index.extend(index_val)
                    max_hit += len(index_val)
                else:
                    break
            return max_hit, mem_index

    def clear(self):
        with self.lock:
            self.radix_buffer.req_mem_index.clear()
            self.lru_queue[:] = []

def build_radix_manager(mem_propties: MemPropties, 
                        use_gpu: bool, 
                        radix_lock) -> RadixBufferManager:
    device = "cuda" if use_gpu else "cpu"

    init_shared_data(
        mem_propties=mem_propties,
        device=device,
    )

    radix_mem_buffer = RadixMemoryBuffer(
        mem_propties=mem_propties,
        shared_data=get_shared_data(),
        lock=radix_lock,
        device=device,
    )

    radix_manager = RadixBufferManager(
        radix_buffer=radix_mem_buffer,
        radix_mem_data=get_shared_data(),
        lock=radix_lock,
    )

    return radix_manager
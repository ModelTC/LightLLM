import ctypes
import torch
import numpy as np
from sortedcontainers import SortedSet
from typing import Optional, List
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.embed_utils import calcu_embed_cache_meta
from lightllm.utils.kv_cache_utils import create_shm_kv_cache_ptr, attach_shm_kv_cache_ptr, register_shm_ptr_to_pin

logger = init_logger(__name__)


class CpuEmbedCacheClient(object):
    """
    This class is responsible for handling cpu kv cache meta data.
    """

    def __init__(self, create_meta_data: bool, init_shm_data: bool):
        self.args = get_env_start_args()
        # to do here need calcu from from settings.
        self.embed_cache_tensor_meta = calcu_embed_cache_meta()
        self.token_num: int = self.embed_cache_tensor_meta.token_num

        if create_meta_data:
            self.token_index_manager = MemoryManager(total_size=self.token_num)
        else:
            if init_shm_data:
                self._create_shm_embed_kv_cache()
            else:
                self._attach_shm_cpu_embed_cache()
        return

    def alloc_indexes(self, token_num: int) -> Optional["MemoryBlock"]:
        return self.token_index_manager.alloc(need_size=token_num)

    def release_indexes(self, block: "MemoryBlock"):
        self.token_index_manager.release(block)
        return

    def copy_to_cache(self, embed_tensor: torch.Tensor, start_index_in_cache: int):
        from .copy_to_cache import offload_embed_tensor_to_cache

        offload_embed_tensor_to_cache(
            embed_tensor=embed_tensor,
            cache_tensor=self.cpu_embed_cache_tensor,
            start_index_in_cache=start_index_in_cache,
        )

    def _create_shm_embed_kv_cache(self):
        shm_ptr = create_shm_kv_cache_ptr(
            key=self.args.multi_modal_cache_shm_id, size=self.embed_cache_tensor_meta.calcu_size()
        )
        handle = register_shm_ptr_to_pin(shm_ptr=shm_ptr, size=self.embed_cache_tensor_meta.calcu_size())
        handle.wait()
        numpy_array = np.frombuffer(
            memoryview((ctypes.c_uint8 * self.embed_cache_tensor_meta.calcu_size()).from_address(shm_ptr)),
            dtype=np.uint8,
        )
        # 将 NumPy 数组转换为 PyTorch 张量
        shape = (
            self.embed_cache_tensor_meta.token_num,
            self.embed_cache_tensor_meta.layer_num,
            self.embed_cache_tensor_meta.hidden_size,
        )
        self.cpu_embed_cache_tensor = (
            torch.from_numpy(numpy_array).view(dtype=self.embed_cache_tensor_meta.data_type).view(shape)
        )
        return

    def _attach_shm_cpu_embed_cache(self):
        shm_ptr = attach_shm_kv_cache_ptr(
            key=self.args.multi_modal_cache_shm_id, size=self.embed_cache_tensor_meta.calcu_size()
        )
        handle = register_shm_ptr_to_pin(shm_ptr=shm_ptr, size=self.embed_cache_tensor_meta.calcu_size())
        handle.wait()
        numpy_array = np.frombuffer(
            memoryview((ctypes.c_uint8 * self.embed_cache_tensor_meta.calcu_size()).from_address(shm_ptr)),
            dtype=np.uint8,
        )
        shape = (
            self.embed_cache_tensor_meta.token_num,
            self.embed_cache_tensor_meta.layer_num,
            self.embed_cache_tensor_meta.hidden_size,
        )
        self.cpu_embed_cache_tensor = (
            torch.from_numpy(numpy_array).view(dtype=self.embed_cache_tensor_meta.data_type).view(shape)
        )
        assert shm_ptr == self.cpu_embed_cache_tensor.data_ptr()
        return None


class MemoryBlock:
    """内存块类，表示一个连续的内存区域"""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def size(self):
        return self.end - self.start

    def __repr__(self):
        return f"Block(start={self.start}, end={self.end})"

    def can_merge(self, block: "MemoryBlock"):
        return (self.start == block.end) or (block.start == self.end)


class MemoryManager:
    def __init__(self, total_size):
        """
        初始化内存管理器
        :param total_size: 总内存大小
        """
        self.total_size = total_size
        self.mem_set_by_start = SortedSet(key=lambda x: (x.start, x.size()))
        self.mem_set_by_size = SortedSet(key=lambda x: (x.size(), x.start))
        total = MemoryBlock(0, total_size)
        self.__add(total)

    def alloc(self, need_size: int) -> Optional[MemoryBlock]:
        assert need_size > 0

        if len(self.mem_set_by_size) == 0:
            return None

        key = MemoryBlock(start=-1, end=-1 + need_size)
        find_index = self.mem_set_by_size.bisect_left(key)
        if find_index < len(self.mem_set_by_size):
            finded_mem_block: MemoryBlock = self.mem_set_by_size[find_index]
            self.__del(finded_mem_block)
            ret_mem_block = MemoryBlock(
                start=finded_mem_block.start,
                end=finded_mem_block.start + need_size,
            )
            left_block = MemoryBlock(
                start=finded_mem_block.start + need_size,
                end=finded_mem_block.end,
            )
            if left_block.size() > 0:
                self.__add(left_block)

            return ret_mem_block
        else:
            return None

    def release(self, block: MemoryBlock):
        if block is None:
            return
        if len(self.mem_set_by_size) == 0:
            self.__add(block)
            return

        finded_index = self.mem_set_by_start.bisect_left(block)
        for index in [finded_index - 1, finded_index, finded_index + 1]:
            if index < len(self.mem_set_by_start):
                sub_block: MemoryBlock = self.mem_set_by_start[index]
                # merge
                if block.can_merge(sub_block):
                    self.__del(sub_block)
                    merge_block = MemoryBlock(
                        start=min(block.start, sub_block.start),
                        end=max(block.end, sub_block.end),
                    )
                    self.release(merge_block)
                    return
        # 无法merge时，直接add
        self.__add(block)
        return

    def __add(self, block):
        self.mem_set_by_start.add(block)
        self.mem_set_by_size.add(block)

    def __del(self, block):
        self.mem_set_by_start.remove(block)
        self.mem_set_by_size.remove(block)


if __name__ == "__main__":
    mem = MemoryManager(total_size=2000)

    import random

    alloced_list = []
    for i in range(20000):
        if random.randint(0, 3) > 1:
            o = mem.alloc(need_size=random.randint(1, 100))
            if o is not None:
                alloced_list.append(o)
        else:
            if len(alloced_list) > 0:
                index = random.randint(0, len(alloced_list) - 1)
                mem.release(alloced_list[index])
                del alloced_list[index]

    for e in alloced_list:
        mem.release(e)

    print(mem.mem_set_by_start)
    assert len(mem.mem_set_by_size) == len(mem.mem_set_by_start)
    assert len(mem.mem_set_by_size) == 1
    print(mem.mem_set_by_size[0])

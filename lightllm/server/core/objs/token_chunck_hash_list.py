import numpy as np
import os
import ctypes
from typing import List
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

LIGHTLLM_TOKEN_HASH_LIST_SIZE = int(os.getenv("LIGHTLLM_TOKEN_HASH_LIST_SIZE", 2048))


class TokenHashList(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("items", ctypes.c_uint64 * (LIGHTLLM_TOKEN_HASH_LIST_SIZE * 2)),  # 存储128位哈希,每个哈希用2个uint64
        ("size", ctypes.c_int),  # 队列大小(表示哈希的数量,非uint64数量)
    ]

    def __init__(self):
        # 初始化头和尾
        self.size = 0
        return

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == LIGHTLLM_TOKEN_HASH_LIST_SIZE

    def fill(self, data: List[int]):
        if len(data) > LIGHTLLM_TOKEN_HASH_LIST_SIZE:
            logger.warning(
                f"Queue capcity is smaller than data size ({len(data)} > {LIGHTLLM_TOKEN_HASH_LIST_SIZE}), "
                f"remove tail to write"
            )
            data = data[0:LIGHTLLM_TOKEN_HASH_LIST_SIZE]
        if len(data) == 0:
            self.size = 0
            return
        hash_array = np.asarray(data, dtype=object)
        low_bits = (hash_array & 0xFFFFFFFFFFFFFFFF).astype(np.uint64)
        high_bits = (hash_array >> 64).astype(np.uint64)
        self.items[0 : len(data) * 2 : 2] = low_bits
        self.items[1 : len(data) * 2 : 2] = high_bits
        self.size = len(data)
        return

    def clear(self):
        self.size = 0

    def get_all(self):
        if self.size == 0:
            return []
        items_array = np.array(self.items[0 : self.size * 2], dtype=np.uint64)
        low = items_array[0::2]
        high = items_array[1::2]
        result = (high.astype(object) << 64) | low.astype(object)
        return result.tolist()


class CpuCachePageList(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("items", ctypes.c_int * LIGHTLLM_TOKEN_HASH_LIST_SIZE),  # 元素静态数组
        ("size", ctypes.c_int),  # 队列大小
    ]

    def __init__(self):
        # 初始化头和尾
        self.size = 0
        return

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == LIGHTLLM_TOKEN_HASH_LIST_SIZE

    def fill(self, data: List[int]):
        assert self.size == 0
        assert len(data) <= LIGHTLLM_TOKEN_HASH_LIST_SIZE
        self.items[0 : len(data)] = data
        self.size = len(data)
        return

    def clear(self):
        self.size = 0

    def get_all(self):
        return list(self.items[0 : self.size])

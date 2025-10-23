import os
import ctypes
from typing import List
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

LIGHTLLM_TOKEN_HASH_LIST_SIZE = int(os.getenv("LIGHTLLM_TOKEN_HASH_LIST_SIZE", 2048))


class TokenHashList(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("items", ctypes.c_uint64 * LIGHTLLM_TOKEN_HASH_LIST_SIZE),  # 元素静态数组
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
        if len(data) > LIGHTLLM_TOKEN_HASH_LIST_SIZE:
            logger.warning(
                f"Queue capcity is smaller than data size ({len(data)} > {LIGHTLLM_TOKEN_HASH_LIST_SIZE}), "
                f"remove tail to write"
            )
            data = data[0:LIGHTLLM_TOKEN_HASH_LIST_SIZE]
        self.items[0 : len(data)] = data
        self.size = len(data)
        return

    def clear(self):
        self.size = 0

    def get_all(self):
        return list(self.items[0 : self.size])


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

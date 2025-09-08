import os
import ctypes
from typing import Tuple

LIGHTLLM_TOKEN_MAX_BYTES = int(os.getenv("LIGHTLLM_TOKEN_MAX_BYTES", 1280))
LIGHTLLM_OUT_TOKEN_QUEUE_SIZE = int(os.getenv("LIGHTLLM_OUT_TOKEN_QUEUE_SIZE", 8))


class QueueItem(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("data", ctypes.c_ubyte * LIGHTLLM_TOKEN_MAX_BYTES),
        ("data_len", ctypes.c_int),
        ("special", ctypes.c_bool),
        ("count_output_tokens", ctypes.c_int),
        ("src_index", ctypes.c_int),  # 在源token队列的索引位置
    ]

    def __init__(self):
        self.data_len = 0
        self.src_index = -1
        self.special = False
        self.count_output_tokens = -1

    def set(self, token_str: str, src_index: int, special: bool, count_output_tokens: int):
        str_bytes = token_str.encode("utf-8")
        assert (
            len(str_bytes) <= LIGHTLLM_TOKEN_MAX_BYTES
        ), f"Token string {len(str_bytes)} exceeds maximum length of {LIGHTLLM_TOKEN_MAX_BYTES} bytes."
        ctypes.memmove(self.data, str_bytes, len(str_bytes))
        self.data_len = len(str_bytes)
        self.src_index = src_index
        self.special = special
        self.count_output_tokens = count_output_tokens
        return

    def get(self):
        return (
            bytes(self.data[: self.data_len]).decode("utf-8"),
            self.src_index,
            self.special,
            self.count_output_tokens,
        )


class CircularQueue(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("items", QueueItem * LIGHTLLM_OUT_TOKEN_QUEUE_SIZE),  # 循环队列的元素
        ("head", ctypes.c_int),  # 指向队列头部
        ("tail", ctypes.c_int),  # 指向队列尾部
    ]

    def __init__(self):
        # 初始化头和尾
        self.head = 0
        self.tail = 0

    def is_empty(self):
        return self.head == self.tail

    def is_full(self):
        return (self.tail + 1) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE == self.head

    def push(self, token_str: str, src_index: int, special: bool, count_output_tokens: int):
        if self.is_full():
            raise Exception("Queue is full")

        # 添加元素
        item: QueueItem = self.items[self.tail]
        item.set(token_str, src_index, special, count_output_tokens)

        # 更新尾部
        self.tail = (self.tail + 1) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE

    def pop(self) -> Tuple[str, int, bool, int]:
        if self.is_empty():
            raise Exception("Queue is empty")

        # 移除元素
        item: QueueItem = self.items[self.head]
        result = item.get()

        # 更新头部
        self.head = (self.head + 1) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE
        return result

    def peek(self) -> Tuple[str, int, bool, int]:
        if self.is_empty():
            raise Exception("Queue is empty")

        item: QueueItem = self.items[self.head]
        result = item.get()
        return result

    def pop_no_ret(self) -> None:
        """
        移除一个对象，但是不返回， 与peek配合使用
        """
        if self.is_empty():
            raise Exception("Queue is empty")
        self.head = (self.head + 1) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE
        return

    def __len__(self):
        # 计算当前元素数量
        return (self.tail - self.head + LIGHTLLM_OUT_TOKEN_QUEUE_SIZE) % LIGHTLLM_OUT_TOKEN_QUEUE_SIZE

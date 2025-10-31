from enum import Enum
from math import ceil
from typing import List

import numpy as np
import torch
import xxhash

from .thext import CompiledExension

class PyState(Enum):
    """ PyState 枚举类型用来描述任务块 (TaskBlock) 的执行状态
    用户需要在任务执行结束后，手动查看每一个任务块的执行状态，
    只有状态为 Finished 的任务块表示任务成功执行，对应的数据有效
    """
    # 任务刚刚完成创建
    Initial = 0
    # 任务正在进行中
    Working = 1
    # 任务成功执行完成
    Finished = 2
    # 任务被终止 (可能因为执行出错)
    Aborted = 3


class PyTask:
    """ PyTask 对象用于描述一个异步存取任务
    你不应该调用这个类型的构造函数，所有 PyTask 都应当由 PyLocalCacheService.create 函数创建
    一旦完成创建，任务便立即会被提交到 PyLocalCacheService 的任务队列并异步地执行

    每一个 PyTask 在底层都会按照 BlockSize 的约定，分解成一系列的 Task Block
    PyLocalCacheService 将以 Block 为单位完成任务的读写

    用户需要在合适的时间调用 ready() 函数查看 PyTask 的执行情况
    如果 ready() 函数返回 True，则说明任务已经执行结束
    此时调用 state() 函数将返回所有任务块的执行情况
    """
    def __init__(self, _c):
        self._c = _c
        self._state_convert = {
            CompiledExension.State.Initial: PyState.Initial,
            CompiledExension.State.Working: PyState.Working,
            CompiledExension.State.Finished: PyState.Finished,
            CompiledExension.State.Aborted: PyState.Aborted
        }

    def ready(self) -> bool:
        """ ready 函数用来询问任务是否完成执行 """
        return self._c.ready()

    def state(self) -> List[PyState]:
        """ state 函数用来询问所有任务块的执行情况 """
        return [self._state_convert[s] for s in self._c.state()]


class PyLocalCacheService:
    """ 基于本地存储的异步数据存取服务 """
    def __init__(
    self, kvcache_tensor: torch.Tensor, file: str, storage_size: int = 32*1024*1024*1024,
    num_shard: int = 32, num_worker: int = 16
    ):
        """ 使用 PyLocalCacheService 来创建异步数据存取引擎 (基于本地磁盘)
    PyLocalCacheService 会直接从 kv cache 中读取数据，并异步地向 kv cache 中写入数据
    kv cache tensor 必须是三维张量，形如 [num of page, num of layer, page size]
    kv cache tensor 必须是连续内存
    kv cache tensor 必须是 CPU 张量

        Args:
            file (str): 本地文件位置
            num_shard(int): 本地文件分片数量
            storage_size (int): 本地文件大小 (本地文件是分片存储的，这里表示总大小)
            kvcache_tensor (torch.Tensor): kvcache tensor，维度顺序为 [page, layer, stride]
            num_worker (int): 工作线程数量
        """
        if kvcache_tensor.dim() != 3:
            raise ValueError("kvcache_tensor 必须是三维张量，形如 [num of page, num of layer, page size]")
        if kvcache_tensor.is_cuda:
            raise ValueError("kvcache_tensor 必须是 CPU 张量，GPU 流程已移除")

        num_pages_total = kvcache_tensor.shape[0]
        num_layers = kvcache_tensor.shape[1]

        self._c = CompiledExension.LocalCacheService(
            file, storage_size, num_shard,
            kvcache_tensor, num_worker
        )
        self._c.run()
        self._num_of_layer: int = num_layers
        self._num_of_page_total: int = num_pages_total
        self._block_size: int = int(self._c.block_size())
        self._page_size: int = int(self._c.page_size())
        self._n: int = ceil(self._block_size / (self._page_size * self._num_of_layer))

    def hash(self, tokens: List[int]) -> List[str]:
        """
        将 tokens 数组按照长度 n 进行划分，并计算哈希值。
        哈希值计算方式：采用递增计算，每次基于上一次的哈希状态更新。

        :param tokens: 整数列表，表示需要哈希的 token 序列。
        :return: 每个分块的哈希值列表。
        """
        ret: List[str] = []

        hsum = xxhash.xxh3_128()
        tokens_np = np.array(tokens, dtype=np.uint32)

        for i in range(0, len(tokens_np), self._n):
            chunk = tokens_np[i: i + self._n].tobytes()
            hsum.update(chunk)
            ret.append(hsum.hexdigest())

        return ret

    def create(
        self, tokens: List[int], kv_page_indexer: torch.Tensor, mode: str, start_pos: int = 0
    ) -> PyTask:
        """ 创建并立即提交一个异步存取任务

        Args:
            tokens (List[int]): 用户传入的 token list
                (实际执行的存取请求面向 tokens[start_pos: ])
                但用户不能够直接传入截断后的 tokens，因为需要 tokens 前面的内容计算哈希
            start_pos (int): 任务开始位置
            kv_page_indexer (torch.Tensor):
                一个索引数组，用来表示 tokens 对应的 kvcache_tensor 位置，必须是一维的
            mode (str): r: 异步读取 | w: 异步写入

        Returns:
            PyTask: 一个任务对象
        """

        # 我们需要根据 start pos, end pos 来确定究竟多少个 block 需要被提交
        # 并且针对真实提交的任务，确定其对应的 kv_page_indexer
        start_block_idx: int = start_pos // self._n
        hashs = self.hash(tokens=tokens)
        hashs = hashs[start_block_idx: ]
        _kv_page_indexer = kv_page_indexer[start_block_idx * self._n: ]

        task = self._c.create(hashs, _kv_page_indexer, mode)
        return PyTask(task)

    def query(self, tokens: List[int]) -> List[bool]:
        """ 查询给定的哈希是否存在(以hash列表形式一次查询多个) """
        hashs = self.hash(tokens=tokens)
        return self._c.query(hashs)

    def get_hash_info(self):
        """获取当前存储的哈希索引信息，以便在不同服务之间共享"""
        return self._c.get_hash_info()

    def set_hash_info(self, hash_info):
        """设置存储的哈希索引信息，与其他服务共享同一份缓存在"""
        self._c.set_hash_info(hash_info)

    def az5(self, t: PyTask):
        """ 终止一个任务的执行，此函数调用后，
        工作线程不会立即停止工作(因为他们是异步的，可能有正在进行的任务)，
        但保证所有工作线程不再会从 kvcache 中读写内容
        
        他为什么叫 AZ5: AZ-5按钮是核反应堆紧急停堆按钮
        按下此按钮会使所有控制棒插入反应堆，停止反应堆运行。
        """
        self._c.az5(t._c)

__all__ = [
    "PyState", "PyTask", "PyLocalCacheService",
]

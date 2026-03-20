import uuid
import threading
import dataclasses
import requests
from typing import Union, Optional
import torch
import time
from collections import deque
import multiprocessing.shared_memory as shm
from ..utils import get_shm_name_data, get_shm_name_embed, free_shm, EmbedRefCountRedis
from .naive_memory_cache import Record, InMemoryCache
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MemoryCacheWithRedis(InMemoryCache):
    def __init__(self, args) -> None:
        super().__init__(args)
        redis_url = f"redis://{args.config_server_host}:{args.redis_port}/0"
        self.redis_cache = EmbedRefCountRedis(
            redis_url=redis_url,
            capacity=args.cache_capacity,
            evict_fraction=args.redis_evict_fraction,
            image_embed_dir=args.image_embed_dir,
        )
        # 这里之所以把cache * 2是因为，在分离模式下，cache 服务只是为了更新redis状态，以及维护图片cache的 token_id
        # 便于 dynamic prompt cache 的使用。所以要把cache_capacity * 2，保障其保留的图片cache > redis 服务维护的
        # 硬盘里的图片image embed 数量。
        self.capacity = max(1, args.cache_capacity * 2)

    # llm 负责release
    def release(self, ids: list[int]) -> None:
        with self.lock:
            for id in ids:
                rec = self._records.get(id)
                if rec is None:
                    continue

                redis_exist = self.redis_cache.query(str(id))
                if redis_exist:
                    self.redis_cache.decr(str(id))

                # remote_vit 模式下 release 可能走“预层提前释放 + 请求结束兜底释放”两条路径，
                # 这里避免本地 ref 被重复减成负数，保证 release 可重复调用。
                if rec.ref > 0:
                    self._update_record_ref(rec, -1)

    # vit 负责set
    def set_items_embed(self, ids: list[int]) -> None:
        with self.lock:
            for id in ids:
                self.redis_cache.insert(str(id))
                rec = self._records.get(id)
                if rec is not None:
                    rec.embed = True
                    if rec.ref > 0:
                        self._update_record_ref_by_id(id, -1)
                # 保留一份 redis 引用，直到真正的消费者读取完成后再 release，
                # 避免 VIT 刚写完文件但 LLM 还没来得及读取时被 LRU 误删。

    def get_items_embed(self, ids: list[int], embeding_only: bool = False) -> list[Optional[bool]]:
        ret = []
        for id in ids:
            if embeding_only:
                exist = self.redis_cache.query(str(id))
            else:
                exist = self.redis_cache.query_and_incre(str(id))
            ret.append(exist)
            if exist:
                rec = self._records.get(id)
                if rec is not None:
                    rec.embed = True
        return ret

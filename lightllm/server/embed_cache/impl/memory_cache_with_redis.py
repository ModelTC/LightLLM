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
        self.cache_capacity = args.cache_capacity * 2

    def release(self, ids: list[int]) -> None:
        with self.lock:
            for id_ in ids:
                self._records[id_].ref -= 1
                self.redis_cache.decr(id_)
                print(self.redis_cache.stats(), flush=True)

    def set_items_embed(self, ids: list[int]) -> None:
        for id in ids:
            self.redis_cache.insert(str(id))

    def get_items_embed(self, ids: list[int]) -> list[Optional[bool]]:
        ret = []
        for id in ids:
            # 避免重复的引用计数增加
            if self._records[id].embed:
                ret.append(True)
                continue
            self._records[id].embed = self.redis_cache.query_and_incre(str(id))
            ret.append(self._records[id].embed)
        return ret

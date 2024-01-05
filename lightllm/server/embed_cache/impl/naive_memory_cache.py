import uuid
import threading
import dataclasses
from ..interface import CacheManager, CacheManagerFactory
from typing import Union
import torch
import time
from collections import deque
import multiprocessing.shared_memory as shm
from ..utils import get_shm_name_data, get_shm_name_embed, free_shm


@dataclasses.dataclass
class Record(object):
    id: int
    md5sum: str
    ref: int
    data: bool
    embed: bool
    createtime: float
    visittime: float
    token_id: int
    token_num: int

@CacheManagerFactory.register("naive")
class InMemoryCache(CacheManager):

    def __init__(self, args) -> None:
        self._records = dict()
        self._md5_to_record = dict()
        self.capacity = max(1, args.cache_capacity)
        self.reserved = max(0, int(self.capacity * args.cache_reserved_ratio))
        self.reserved = min(self.reserved, self.capacity - 1)
        self.occupied = 0
        self.expired_secs = 60 * 60
        self.lock = threading.Lock()

        from lightllm.server.tokenizer import get_tokenizer
        tokenizer = get_tokenizer(
            args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code
        )
        self.cur_token_id = tokenizer.vocab_size + 10000

    def _clear(self):
        deleted = 0
        max_delete = max(1, self.occupied - self.reserved)
        items = sorted(self._records.items(), key=lambda x: x[1].visittime)
        t = time.time()
        for id, record in items:
            if record.ref <= 0 or t - record.visittime >= self.expired_secs:
                if record.data:
                    free_shm(get_shm_name_data(id))
                if record.embed:
                    free_shm(get_shm_name_embed(id))
                del self._md5_to_record[record.md5sum]
                del self._records[id]
                self.occupied -= 1
                deleted += 1
                if deleted >= max_delete:
                    break

    def alloc(self, md5sum: str, token_num: int) -> dict:
        with self.lock:
            t = time.time()
            # add new record
            if md5sum not in self._md5_to_record:

                # full, need to clear some unused items
                if self.occupied >= self.capacity:
                    self._clear()
                    if self.occupied >= self.capacity:
                        return None

                id = uuid.uuid1()
                id = id.int
                record = Record(
                    id=id,
                    md5sum=md5sum,
                    ref=1,
                    data=False,
                    embed=False,
                    createtime=t,
                    visittime=t,
                    token_id=self.cur_token_id,
                    token_num=token_num,
                )
                self.cur_token_id += token_num
                self._records[id] = record
                self._md5_to_record[md5sum] = record
                self.occupied += 1

            # cache hit
            else:
                record = self._md5_to_record[md5sum]
                record.visittime = t
                record.ref += 1

            return {
                "id": record.id,
                "token_id": record.token_id,
                "token_num": record.token_num
            }

    def release(self, id: int) -> None:
        with self.lock:
            self._records[id].ref -= 1

    def set_item_data(self, id: int) -> None:
        self._records[id].data = True

    def get_item_data(self, id: int) -> bool:
        return self._records[id].data

    def set_item_embed(self, id: int) -> None:
        self._records[id].embed = True

    def get_item_embed(self, id: int) -> bool:
        return self._records[id].embed

import uuid
import threading
import dataclasses
import requests
from ..interface import CacheManager, CacheManagerFactory
from typing import Union
import torch
import time
from collections import deque
import multiprocessing.shared_memory as shm
from ..utils import get_shm_name_data, get_shm_name_embed, free_shm
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


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
        self.args = args
        self._records = dict()
        self._md5_to_record = dict()
        self.capacity = max(1, args.cache_capacity)
        self.reserved = max(0, int(self.capacity * args.cache_reserved_ratio))
        self.reserved = min(self.reserved, self.capacity - 1)
        self.occupied = 0
        self.expired_secs = 60 * 60
        self.lock = threading.Lock()
        self.token_id_range_start = 0
        self.token_id_range_end = 0
        self.use_config_server = self.args.config_server_host and self.args.config_server_port

    def _check_and_set_new_id_range(self, alloced_token_num):
        need_update_range = self.token_id_range_start + alloced_token_num >= self.token_id_range_end
        if need_update_range:
            if not self.use_config_server:
                self.token_id_range_start = 100000000
                self.token_id_range_end = 2 ** 63 - 1
            else:
                while True:
                    try:
                        config_server_ip_port = f"{self.args.config_server_host}:{self.args.config_server_port}"
                        url = f"http://{config_server_ip_port}/allocate_global_unique_multimodal_id_range"
                        response = requests.get(url)
                        if response.status_code == 200:
                            id_range = response.json()
                            logger.info(f"get new multimodal id range {id_range}")
                            self.token_id_range_start = id_range["start_id"]
                            self.token_id_range_end = id_range["end_id"]
                            assert (
                                self.token_id_range_start + alloced_token_num < self.token_id_range_end
                            ), f"get multimodal id range error {self.token_id_range_start} {self.token_id_range_end}"
                            return
                        else:
                            raise RuntimeError(f"Failed to fetch ID range from config server: {response.status_code}")
                    except BaseException as e:
                        logger.exception(str(e))
                        time.sleep(3)
        return

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
                self._check_and_set_new_id_range(token_num)
                record = Record(
                    id=id,
                    md5sum=md5sum,
                    ref=1,
                    data=False,
                    embed=False,
                    createtime=t,
                    visittime=t,
                    token_id=self.token_id_range_start,
                    token_num=token_num,
                )
                self.token_id_range_start += token_num
                self._records[id] = record
                self._md5_to_record[md5sum] = record
                self.occupied += 1

            # cache hit
            else:
                record = self._md5_to_record[md5sum]
                record.visittime = t
                record.ref += 1

            return {"id": record.id, "token_id": record.token_id, "token_num": record.token_num}

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

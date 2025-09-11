import uuid
import threading
import dataclasses
import requests
from typing import Union, Optional
import torch
import time
from collections import deque
import multiprocessing.shared_memory as shm
from ..utils import get_shm_name_data, get_shm_name_embed, free_shm, free_afs
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


class InMemoryCache:
    def __init__(self, args) -> None:
        self.args = args
        self._records = dict()
        self._md5_to_record = dict()
        self.capacity = max(1, args.cache_capacity)
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

    def _clear(self, free_max_count: int):
        deleted = 0
        max_delete = free_max_count
        items = sorted(self._records.items(), key=lambda x: x[1].visittime)
        t = time.time()
        for id, record in items:
            if record.ref <= 0 or t - record.visittime >= self.expired_secs:
                if record.data:
                    free_shm(get_shm_name_data(id))
                if record.embed:
                    # 仅vit释放掉afs里的, llm端不做释放
                    # if self.args.run_mode == "visual":
                    #     free_afs(get_shm_name_embed(id), self.args.image_embed_dir)
                    # elif not self.args.enable_remote_vit:
                    if not self.args.run_mode == "visual":
                        free_shm(get_shm_name_embed(id))
                del self._md5_to_record[record.md5sum]
                del self._records[id]
                self.occupied -= 1
                deleted += 1
                if deleted >= max_delete:
                    break

    def alloc(self, md5sum_list: list[str], token_num_list: list[int]) -> Optional[list[dict]]:
        now = time.time()
        with self.lock:
            new_md5s = [m for m in md5sum_list if m not in self._md5_to_record]
            new_needed = len(set(new_md5s))

            if self.occupied + new_needed > self.capacity:
                self._clear(free_max_count=new_needed - (self.capacity - self.occupied))
            if self.occupied + new_needed > self.capacity:
                return None

            results: list[dict] = []
            for md5sum, token_num in zip(md5sum_list, token_num_list):
                if md5sum in self._md5_to_record:
                    rec = self._md5_to_record[md5sum]
                    rec.visittime = now
                    rec.ref += 1
                else:
                    uid_int = md5sum
                    self._check_and_set_new_id_range(token_num)
                    rec = Record(
                        id=uid_int,
                        md5sum=md5sum,
                        ref=1,
                        data=False,
                        embed=False,
                        createtime=now,
                        visittime=now,
                        token_id=self.token_id_range_start,
                        token_num=token_num,
                    )
                    self.token_id_range_start += token_num
                    self._records[uid_int] = rec
                    self._md5_to_record[md5sum] = rec
                    self.occupied += 1
                results.append({"id": rec.id, "token_id": rec.token_id, "token_num": rec.token_num})
        return results

    def release(self, ids: list[int]) -> None:
        with self.lock:
            for id_ in ids:
                self._records[id_].ref -= 1

    def set_items_data(self, ids: list[int]) -> None:
        for id_ in ids:
            self._records[id_].data = True

    def get_items_data(self, ids: list[int]) -> list[Optional[bool]]:
        return [self._records.get(id_).data if id_ in self._records else False for id_ in ids]

    def set_items_embed(self, ids: list[int]) -> None:
        for id_ in ids:
            self._records[id_].embed = True

    def get_items_embed(self, ids: list[int]) -> list[Optional[bool]]:
        return [self._records.get(id_).embed if id_ in self._records else False for id_ in ids]

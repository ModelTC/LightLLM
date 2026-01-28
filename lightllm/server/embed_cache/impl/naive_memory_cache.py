import uuid
import threading
import dataclasses
import requests
from typing import Union, Optional
import torch
import time
from collections import deque
from typing import List, Dict
import multiprocessing.shared_memory as shm
from ..utils import get_shm_name_data, free_shm
from lightllm.utils.log_utils import init_logger
from ..embed_cache_client import CpuEmbedCacheClient, MemoryBlock, SortedSet

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
    mem_block: MemoryBlock

    def __hash__(self) -> int:
        return self.id


class InMemoryCache:
    def __init__(self, args) -> None:
        self.args = args
        self._id_to_records = dict()
        self._md5_to_record = dict()
        self._sorted_records = SortedSet(key=lambda x: (x.ref, x.visittime, x.id))
        self.capacity = max(1, args.cache_capacity)
        self.occupied = 0
        self.expired_secs = 60 * 60
        self.lock = threading.Lock()
        self.token_id_range_start = 0
        self.token_id_range_end = 0
        self.use_config_server = self.args.config_server_host and self.args.config_server_port
        self.cpu_embed_cache_client = CpuEmbedCacheClient(create_meta_data=True, init_shm_data=False)

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

    def _try_free_one(self) -> bool:
        if len(self._sorted_records) > 1 and self._sorted_records[0].ref <= 0:
            record: Record = self._sorted_records[0]
            if record.data:
                free_shm(get_shm_name_data(record.id))

            self.cpu_embed_cache_client.release_indexes(block=record.mem_block)
            del self._md5_to_record[record.md5sum]
            del self._id_to_records[record.id]
            del self._sorted_records[0]
            self.occupied -= 1
            return True
        return False

    def _free_to_alloc(self, free_min_count: int, new_md5_dict: Dict[str, int]) -> Dict[str, MemoryBlock]:
        deleted = 0
        while free_min_count > 0 and self._try_free_one():
            deleted += 1
            if deleted >= free_min_count:
                break

        if deleted < free_min_count:
            return {}

        alloc_md5_dict = {}
        for md5, token_alloc in new_md5_dict.items():
            alloc_mem_block = self.cpu_embed_cache_client.alloc_indexes(token_num=token_alloc)
            if alloc_mem_block is not None:
                alloc_md5_dict[md5] = alloc_mem_block
            else:
                _alloc_mem_block = None
                while self._try_free_one():
                    _alloc_mem_block = self.cpu_embed_cache_client.alloc_indexes(token_num=token_alloc)
                    if _alloc_mem_block is not None:
                        break
                if _alloc_mem_block is not None:
                    alloc_md5_dict[md5] = _alloc_mem_block
                else:
                    break

        if len(alloc_md5_dict) != len(new_md5_dict):
            # 放弃分配并放回
            for block in alloc_md5_dict.values():
                self.cpu_embed_cache_client.release_indexes(block)
            return {}
        else:
            return alloc_md5_dict

    def _add_ref(self, md5_sum):
        rec: Record = self._md5_to_record[md5_sum]
        self._sorted_records.remove(rec)
        rec.ref += 1
        self._sorted_records.add(rec)
        return

    def _del_ref(self, md5_sum):
        rec: Record = self._md5_to_record[md5_sum]
        self._sorted_records.remove(rec)
        rec.ref -= 1
        self._sorted_records.add(rec)
        return

    def _judge_enough_token_cache(self, md5sum_list: list[str], token_num_list: list[int]) -> bool:
        tmp_dict = {}
        for md5, token_num in zip(md5sum_list, token_num_list):
            tmp_dict[md5] = token_num
        return sum(tmp_dict.values()) < self.cpu_embed_cache_client.token_num / 3

    def alloc(self, md5sum_list: list[str], token_num_list: list[int]) -> Optional[list[dict]]:
        now = time.time()
        with self.lock:
            if not self._judge_enough_token_cache(md5sum_list=md5sum_list, token_num_list=token_num_list):
                return "error not enough embed cache"

            add_ref_m_list = []
            new_md5_dict = {}
            for m, token_need in zip(md5sum_list, token_num_list):
                if m in self._md5_to_record:
                    # 锁定
                    self._add_ref(m)
                    add_ref_m_list.append(m)
                else:
                    new_md5_dict[m] = token_need

            new_needed = len(new_md5_dict)

            alloc_md5_dict = self._free_to_alloc(
                free_min_count=new_needed - (self.capacity - self.occupied), new_md5_dict=new_md5_dict
            )
            if len(alloc_md5_dict) == len(new_md5_dict):
                for md5sum, mem_block in alloc_md5_dict.items():
                    token_num = new_md5_dict[md5sum]
                    uid_int = uuid.uuid1().int
                    self._check_and_set_new_id_range(token_num)
                    rec = Record(
                        id=uid_int,
                        md5sum=md5sum,
                        ref=0,
                        data=False,
                        embed=False,
                        createtime=now,
                        visittime=now,
                        token_id=self.token_id_range_start,
                        token_num=token_num,
                        mem_block=mem_block,
                    )
                    self.token_id_range_start += token_num
                    self._id_to_records[uid_int] = rec
                    self._md5_to_record[md5sum] = rec
                    self._sorted_records.add(rec)
                    self.occupied += 1

                for md5 in add_ref_m_list:
                    # 解锁
                    self._del_ref(md5)

                # 遍历加 ref
                results = []
                for md5 in md5sum_list:
                    self._add_ref(md5)
                    rec: Record = self._md5_to_record[md5]
                    results.append(
                        {
                            "id": rec.id,
                            "token_id": rec.token_id,
                            "start_index_in_embed_cache": rec.mem_block.start,
                            "token_num": rec.token_num,
                        }
                    )

                return results
            else:
                return None

    def release(self, ids: list[int]) -> None:
        with self.lock:
            for id_ in ids:
                rec: Record = self._id_to_records[id_]
                self._sorted_records.remove(rec)
                rec.ref -= 1
                self._sorted_records.add(rec)

    def set_items_data(self, ids: list[int]) -> None:
        for id_ in ids:
            self._id_to_records[id_].data = True

    def get_items_data(self, ids: list[int]) -> list[Optional[bool]]:
        return [self._id_to_records.get(id_).data if id_ in self._id_to_records else False for id_ in ids]

    def set_items_embed(self, ids: list[int]) -> None:
        for id_ in ids:
            self._id_to_records[id_].embed = True

    def get_items_embed(self, ids: list[int]) -> list[Optional[bool]]:
        return [self._id_to_records.get(id_).embed if id_ in self._id_to_records else False for id_ in ids]

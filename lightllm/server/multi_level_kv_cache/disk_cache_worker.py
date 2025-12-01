import os
import tempfile
import time
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger
from .cpu_cache_client import CpuKvCacheClient

logger = init_logger(__name__)

try:
    from light_mem import PyLocalCacheService, PyState
except ImportError as e:
    logger.error(
        "Failed to import LightMem library. Please install it first.\n"
        "You can install it by building from source: https://github.com/ModelTC/LightMem"
    )
    raise ImportError("LightMem library is required for disk cache functionality") from e


@dataclass
class _PagePayload:
    index: int
    hash_key: int


class DiskCacheWorker:
    """Background worker that offloads CPU KV pages to disk using kvcache."""

    def __init__(
        self,
        disk_cache_storage_size: float,
        cpu_cache_client: CpuKvCacheClient,
        disk_cache_dir: Optional[str] = None,
    ):
        self.cpu_cache_client = cpu_cache_client
        self._pages_all_idle = False

        assert disk_cache_storage_size > 0
        storage_size = int(disk_cache_storage_size * (1024 ** 3))
        # num_shard与KVCACHE_MAX_BLOCK_SIZE相关，KVCACHE_MAX_BLOCK_SIZE默认64MB前提下，
        # num_shard设置32, 能使disk cache的容量利用率达到90%，继续增大num_shard会导致容量利用率下降
        num_shard = 32
        num_worker = 48
        # 读写同时进行时，分配16线程用来写，32线程用来读
        max_concurrent_write_tasks = 16

        cache_dir = disk_cache_dir
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), f"lightllm_disk_cache_{get_unique_server_name()}")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "cache_file")

        self.max_concurrent_write_tasks = max_concurrent_write_tasks
        self._page_major_tensor = self._prepare_tensor(cpu_cache_client.cpu_kv_cache_tensor)

        self.service = PyLocalCacheService(
            kvcache_tensor=self._page_major_tensor,
            file=cache_file,
            storage_size=storage_size,
            num_shard=num_shard,
            num_worker=num_worker,
        )

        logger.info(
            "disk cache worker initialized: dir=%s size_bytes=%d shards=%d workers=%d pages_per_block=%d",
            cache_dir,
            storage_size,
            num_shard,
            num_worker,
            self.service._n,
        )

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.flatten(1).view(dtype=torch.uint8)

    def run(self) -> None:
        while True:
            time.sleep(0.1)
            payload_groups = self._gather_offload_payloads()
            if not payload_groups:
                continue
            for payloads in payload_groups:
                if not payloads:
                    continue
                self._persist_pages_to_disk(payloads)

    def _gather_offload_payloads(self) -> List[List[_PagePayload]]:
        self.cpu_cache_client.lock.acquire_sleep1ms()
        grouped_indexes = self.cpu_cache_client.get_pages_to_offloading()
        self.cpu_cache_client.lock.release()

        payload_groups: List[List[_PagePayload]] = []
        if not grouped_indexes:
            return payload_groups

        page_items = self.cpu_cache_client.page_items.linked_items
        for group in grouped_indexes:
            payloads: List[_PagePayload] = []
            for page_index in group:
                page_item = page_items[page_index]
                payloads.append(_PagePayload(index=page_index, hash_key=int(page_item.hash_key)))
            payload_groups.append(payloads)

        return payload_groups

    # 数据写入磁盘
    def _persist_pages_to_disk(self, payloads: List[_PagePayload]) -> None:
        if not payloads:
            return
        page_indexes = [payload.index for payload in payloads]
        tokens = [payload.hash_key for payload in payloads]
        if not page_indexes:
            return

        kv_indexer = torch.tensor(page_indexes, dtype=torch.int32, device="cpu")
        query_result = self.service.query(tokens)
        if not all(query_result):
            # 限制写入并发量，给读取操作留资源
            while (
                self.service.active_threads("r") and self.service.active_threads("w") >= self.max_concurrent_write_tasks
            ):
                time.sleep(0.001)

            task = self.service.create(tokens=tokens, kv_page_indexer=kv_indexer, mode="w")
            # 立即释放已经在disk cache中的页面
            if task.page_already_list:
                self.cpu_cache_client.lock.acquire_sleep1ms()
                self.cpu_cache_client.deref_pages(page_list=task.page_already_list)
                self.cpu_cache_client.lock.release()

            # 数据安全即可结束等待，无需写入完成
            while not task.data_safe():
                time.sleep(0.001)

            # 释放剩余需要写入的页面
            remining_indexes = list(set(page_indexes) - set(task.page_already_list))
            if remining_indexes:
                self.cpu_cache_client.lock.acquire_sleep1ms()
                self.cpu_cache_client.deref_pages(page_list=remining_indexes)
                self.cpu_cache_client.lock.release()
        else:
            self.cpu_cache_client.lock.acquire_sleep1ms()
            self.cpu_cache_client.deref_pages(page_list=page_indexes)
            self.cpu_cache_client.lock.release()

    def query_loadable_pages(self, tokens: List[int], start_pos: int) -> int:
        """
        查询从start_pos位置开始,可以从disk cache加载的最长前缀长度
        Returns:
            loadable_len: 从start_pos开始可以加载的长度
        """
        if not tokens or start_pos < 0 or start_pos >= len(tokens):
            return 0

        query_result = self.service.query(tokens)
        n = self.service._n
        start_block = start_pos // n
        try:
            first_false_idx = start_block + query_result[start_block:].index(False)
        except ValueError:
            return len(tokens) - start_pos
        first_missing_pos = first_false_idx * n
        return max(0, first_missing_pos - start_pos)

    # 从磁盘读取数据到内存
    def load_pages(self, tokens: List[int], page_indexes: List[int], start_pos: int = 0) -> bool:
        if not tokens or not page_indexes or len(tokens) != len(page_indexes):
            return False
        if start_pos < 0 or start_pos >= len(tokens):
            return False

        # 检测当前是否有写操作在进行，若有则跳过本次load请求，暂时不用
        # if self.service.active_threads("w") > 0:
        #     logger.warning("disk cache worker is busy writing, skip load_pages")
        #     return False

        kv_indexer = torch.tensor(page_indexes, dtype=torch.int32, device="cpu")
        task = self.service.create(tokens=tokens, kv_page_indexer=kv_indexer, mode="r", start_pos=start_pos)
        while not task.ready():
            time.sleep(0.001)
        return all(state == PyState.Finished for state in task.state())

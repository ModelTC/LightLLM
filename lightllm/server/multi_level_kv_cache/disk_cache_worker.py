import os
import tempfile
import time
import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from cache import PyLocalCacheService, PyState
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@dataclass
class _PagePayload:
    index: int
    hash_key: int


class DiskCacheWorker:
    """Background worker that offloads CPU KV pages to disk using kvcache."""

    def __init__(self, disk_cache_storage_size: float, cpu_cache_client):
        self.cpu_cache_client = cpu_cache_client
        self._pages_all_idle = False

        assert disk_cache_storage_size > 0
        storage_size = int(disk_cache_storage_size * (1024 ** 3))
        num_shard = 32
        num_worker = 32

        cache_dir = os.getenv("LIGHTLLM_DISK_CACHE_DIR")
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), f"lightllm_disk_cache_{get_unique_server_name()}")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "cache_file")

        self._page_major_tensor = self._prepare_tensor(cpu_cache_client.cpu_kv_cache_tensor)

        self.service = PyLocalCacheService(
            kvcache_tensor=self._page_major_tensor,
            file=cache_file,
            storage_size=storage_size,
            num_shard=num_shard,
            num_worker=num_worker,
        )

        logger.info(
            "blueswhen disk cache worker initialized: dir=%s size_bytes=%d shards=%d workers=%d pages_per_block=%d",
            cache_dir,
            storage_size,
            num_shard,
            num_worker,
            self.service._n,
        )

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        num_page, num_layer = tensor.shape[0], tensor.shape[1]
        return tensor.reshape(num_page, num_layer, -1)

    def run(self) -> None:
        while True:
            time.sleep(0.01)
            payload_groups = self._gather_offload_payloads()
            # self._log_idle_once()
            if not payload_groups:
                continue
            for payloads in payload_groups:
                if not payloads:
                    continue
                self._persist_pages_to_disk(payloads)

    def _log_idle_once(self) -> int:
        locked_pages = 0
        self.cpu_cache_client.lock.acquire_sleep1ms()
        try:
            for page_idx in range(self.cpu_cache_client.page_num):
                page_item = self.cpu_cache_client.page_items.get_item_by_index(page_idx)
                if not page_item.is_ready_recycle() or page_item.ref_count != 0:
                    locked_pages += 1
        finally:
            self.cpu_cache_client.lock.release()

        if locked_pages == 0:
            if not self._pages_all_idle:
                logger.info("blueswhen all cpu cache pages are idle and ready to reuse")
            self._pages_all_idle = True
        else:
            self._pages_all_idle = False
        return locked_pages

    def _gather_offload_payloads(self) -> List[List[_PagePayload]]:
        self.cpu_cache_client.lock.acquire_sleep1ms()
        try:
            grouped_indexes = self.cpu_cache_client.get_pages_to_offloading()
            payload_groups: List[List[_PagePayload]] = []
            if not grouped_indexes:
                return payload_groups
            for group in grouped_indexes:
                payloads: List[_PagePayload] = []
                for page_index in group:
                    page_item = self.cpu_cache_client.page_items.get_item_by_index(page_index)
                    payloads.append(_PagePayload(index=page_index, hash_key=int(page_item.hash_key)))
                payload_groups.append(payloads)
            return payload_groups
        finally:
            self.cpu_cache_client.lock.release()

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
            task = self.service.create(tokens=tokens, kv_page_indexer=kv_indexer, mode="w")
            while not task.ready():
                time.sleep(0.001)

        self.cpu_cache_client.lock.acquire_sleep1ms()
        self.cpu_cache_client.update_pages_status_to_ready_recycle(page_list=page_indexes, deref=True)
        self.cpu_cache_client.lock.release()

        # self._log_idle_once()

    def blocks_exist(self, tokens: List[int], start_pos: int = 0) -> bool:
        if not tokens or start_pos < 0 or start_pos >= len(tokens):
            return False

        query_result = self.service.query(tokens)
        block_start = start_pos // self.service._n
        block_end = math.ceil(len(tokens) / self.service._n)
        if block_start >= block_end:
            return False
        return all(query_result[block_start:block_end])

    def load_pages(self, tokens: List[int], page_indexes: List[int], start_pos: int = 0) -> bool:
        if not tokens or not page_indexes or len(tokens) != len(page_indexes):
            return False
        if start_pos < 0 or start_pos >= len(tokens):
            return False

        kv_indexer = torch.tensor(page_indexes, dtype=torch.int32, device="cpu")
        task = self.service.create(tokens=tokens, kv_page_indexer=kv_indexer, mode="r", start_pos=start_pos)
        while not task.ready():
            time.sleep(0.001)
        return all(state == PyState.Finished for state in task.state())

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


@dataclass
class _PagePayload:
    index: int
    hash_key: int


class DiskCacheWorker:
    """Background worker that offloads CPU KV pages to disk using kvcache."""

    def __init__(self, args, cpu_cache_client):
        self.logger = init_logger(__name__ + ".disk")
        self.args = args
        self.cpu_cache_client = cpu_cache_client
        self.enabled = bool(getattr(args, "enable_disk_cache", False))
        self._idle_sleep = float(os.getenv("LIGHTLLM_DISK_CACHE_IDLE_SLEEP", "0.02"))
        self._stop = False
        self._last_all_idle = False
        self._last_locked_count = -1

        self.service: Optional[PyLocalCacheService] = None
        self._page_major_tensor: Optional[torch.Tensor] = None

        if not self.enabled:
            self.logger.info("disk cache disabled by configuration")
            return

        storage_size = max(int(args.disk_cache_storage_size * (1024**3)), 1)
        num_shard = max(int(os.getenv("LIGHTLLM_DISK_CACHE_SHARDS", "32")), 1)
        num_worker = max(int(os.getenv("LIGHTLLM_DISK_CACHE_WORKERS", "32")), 1)

        cache_dir = os.getenv("LIGHTLLM_DISK_CACHE_DIR")
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), f"lightllm_disk_cache_{get_unique_server_name()}")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "cache_file")

        page_major_tensor = self._prepare_tensor(cpu_cache_client.cpu_kv_cache_tensor)

        self.service = PyLocalCacheService(
            kvcache_tensor=page_major_tensor,
            file=cache_file,
            storage_size=storage_size,
            num_shard=num_shard,
            num_worker=num_worker,
        )
        self._page_major_tensor = page_major_tensor

        self.logger.info(
            "disk cache worker initialized: dir=%s size_bytes=%d shards=%d workers=%d pages_per_block=%d",
            cache_dir,
            storage_size,
            num_shard,
            num_worker,
            self.service._n,
        )

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            return tensor
        if tensor.dim() == 5:
            num_page, num_layer = tensor.shape[0], tensor.shape[1]
            return tensor.reshape(num_page, num_layer, -1)
        raise ValueError(f"Unsupported kv cache tensor shape: {tuple(tensor.shape)}")

    def run(self) -> None:
        if not self.enabled or self.service is None:
            return

        while not self._stop:
            payload_groups = self._claim_ready_pages()
            self._log_idle_once()
            if not payload_groups:
                time.sleep(self._idle_sleep)
                continue
            try:
                for payloads in payload_groups:
                    if not payloads:
                        continue
                    self._flush_pages(payloads)
            except Exception:
                self.logger.exception("disk cache write failed, restoring pages to READY state")
                for payloads in payload_groups:
                    if payloads:
                        self._restore_pages(payloads)
                time.sleep(self._idle_sleep)

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
            if not self._last_all_idle:
                self.logger.info("blueswhen all cpu cache pages are idle and ready to reuse")
            self._last_all_idle = True
        else:
            self._last_all_idle = False
        return locked_pages

    def _log_locked_pages_count(self, locked_pages: int) -> None:
        if locked_pages < 0:
            return
        if locked_pages == 0:
            self._last_locked_count = 0
            self._last_all_idle = True
        else:
            if locked_pages != self._last_locked_count:
                # self.logger.info("blueswhen %d cpu cache pages still locked", locked_pages)
                self._last_locked_count = locked_pages
            else:
                self._last_locked_count = locked_pages

    def _claim_ready_pages(self) -> List[List[_PagePayload]]:
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

    def _flush_pages(self, payloads: List[_PagePayload]) -> None:
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
        try:
            self.cpu_cache_client.update_pages_status_to_ready_recycle(page_list=page_indexes, deref=True)
        finally:
            self.cpu_cache_client.lock.release()

        # After completing a flush, re-check idle state immediately and
        # emit diagnostic info if not all pages are idle. This helps
        # distinguish between a timing/log-order issue (log checked
        # before flush) and real leftover references.
        locked_pages = self._log_idle_once()
        self._log_locked_pages_count(locked_pages)

    def _restore_pages(self, payloads: List[_PagePayload]) -> None:
        if not payloads:
            return

        self.cpu_cache_client.lock.acquire_sleep1ms()
        try:
            for payload in payloads:
                page_item = self.cpu_cache_client.page_items.get_item_by_index(payload.index)
                if page_item.status == page_item.OFFLOADING:
                    page_item.status = page_item.READY
                    if page_item.ref_count > 0:
                        page_item.ref_count -= 1
                    self.cpu_cache_client.offload_page_indexes.add_item(payload.index)
        finally:
            self.cpu_cache_client.lock.release()

    def stop(self) -> None:
        self._stop = True

    def blocks_exist(self, tokens: List[int], start_pos: int = 0) -> bool:
        if not self.enabled or self.service is None or not tokens:
            return False
        if start_pos < 0 or start_pos >= len(tokens):
            return False

        query_result = self.service.query(tokens)
        block_start = start_pos // self.service._n
        block_end = math.ceil(len(tokens) / self.service._n)
        if block_start >= block_end:
            return False
        return all(query_result[block_start:block_end])

    def load_pages(self, tokens: List[int], page_indexes: List[int], start_pos: int = 0) -> bool:
        if not self.enabled or self.service is None:
            return False
        if not tokens or not page_indexes or len(tokens) != len(page_indexes):
            return False
        if start_pos < 0 or start_pos >= len(tokens):
            return False

        kv_indexer = torch.tensor(page_indexes, dtype=torch.int32, device="cpu")
        task = self.service.create(tokens=tokens, kv_page_indexer=kv_indexer, mode="r", start_pos=start_pos)
        try:
            while not task.ready():
                time.sleep(0.001)
        except Exception:
            return False
        return all(state == PyState.Finished for state in task.state())
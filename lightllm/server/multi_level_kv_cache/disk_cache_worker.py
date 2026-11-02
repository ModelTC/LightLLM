import os
import tempfile
import time
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
from lightllm.common.attention_state.lifecycle import UncertainCompletion
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

TASK_WAIT_TIMEOUT_S = 120.0


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
        state_manifest=None,
    ):
        self.cpu_cache_client = cpu_cache_client
        self._pages_all_idle = False
        self._failed = False
        self._retired_tasks = []

        assert disk_cache_storage_size > 0
        storage_size = int(disk_cache_storage_size * (1024 ** 3))
        # 为了发挥afs性能，2TiB以内使用8个shard；容量超过2TiB后，每增加1TiB增加4个shard。
        num_shard = max(8, math.ceil(disk_cache_storage_size / 1024) * 4)
        num_worker = 24
        # 读写同时进行时，分配8线程用来写，16线程用来读
        max_concurrent_write_tasks = 8

        cache_dir = disk_cache_dir
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), f"lightllm_disk_cache_{get_unique_server_name()}")
        os.makedirs(cache_dir, exist_ok=True)
        if state_manifest is not None:
            from lightllm.common.attention_state.manifest import validate_disk_manifest

            validate_disk_manifest(cache_dir, state_manifest)
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

    def _wait_task(self, task, cond_name: str) -> bool:
        deadline = time.monotonic() + TASK_WAIT_TIMEOUT_S
        while True:
            try:
                if getattr(task, cond_name)():
                    return True
            except Exception as error:
                self._retain_failed_task(task)
                raise UncertainCompletion("disk completion query failed") from error
            if time.monotonic() >= deadline:
                logger.error(
                    f"disk cache task '{cond_name}' wait timeout after "
                    f"{TASK_WAIT_TIMEOUT_S:.1f}s, aborting task to avoid hang"
                )
                try:
                    self.service.abort(task)
                except Exception as e:
                    self._retain_failed_task(task)
                    raise UncertainCompletion("disk cache abort failed; pages must remain reserved") from e
                return False
            time.sleep(0.001)

    def _submit_task(self, **kwargs):
        try:
            return self.service.create(**kwargs)
        except Exception as error:
            self._retain_failed_task(None)
            raise UncertainCompletion("disk submission failed without a completion handle") from error

    def _retain_failed_task(self, task):
        self._failed = True
        if not hasattr(self, "_retired_tasks"):
            self._retired_tasks = []
        self._retired_tasks.append(task)

    def _quarantine_pages(self, pages):
        self.cpu_cache_client.lock.acquire_sleep1ms()
        try:
            self.cpu_cache_client.quarantine_pages(pages)
        finally:
            self.cpu_cache_client.lock.release()

    def _release_pages(self, pages):
        self.cpu_cache_client.lock.acquire_sleep1ms()
        try:
            self.cpu_cache_client.deref_pages(pages)
        finally:
            self.cpu_cache_client.lock.release()

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
        hashs = [payload.hash_key for payload in payloads]
        if not page_indexes:
            return
        if getattr(self, "_failed", False):
            self._release_pages(page_indexes)
            return

        kv_indexer = torch.tensor(page_indexes, dtype=torch.int32, device="cpu")
        try:
            query_result = self.service.query(hashs)
            if not all(query_result):
                # 限制写入并发量，给读取操作留资源
                throttle_deadline = time.monotonic() + TASK_WAIT_TIMEOUT_S
                while (
                    self.service.active_threads("r")
                    and self.service.active_threads("w") >= self.max_concurrent_write_tasks
                ):
                    if time.monotonic() >= throttle_deadline:
                        logger.error(
                            f"disk cache write throttle wait timeout after {TASK_WAIT_TIMEOUT_S:.1f}s, "
                            "proceeding to submit"
                        )
                        break
                    time.sleep(0.001)
        except Exception:

            self._failed = True
            self._release_pages(page_indexes)
            logger.exception("Disk query failed before submission; disabling disk I/O")
            return

        if not all(query_result):
            try:
                task = self._submit_task(hash_128s=hashs, kv_page_indexer=kv_indexer, mode="w")
            except UncertainCompletion:
                self._quarantine_pages(page_indexes)
                logger.exception("Disk submission failed; retaining its source pages")
                return
            # 立即释放已经在disk cache中的页面
            try:
                already_present = task.page_already_list
            except Exception:
                self._retain_failed_task(task)
                self._quarantine_pages(page_indexes)
                logger.exception("Disk submission status is unknown; retaining its source pages")
                return
            if already_present:
                self._release_pages(already_present)

            remaining_indexes = list(set(page_indexes) - set(already_present))
            try:
                self._wait_task(task, "data_safe")
            except UncertainCompletion:
                self._quarantine_pages(remaining_indexes)
                logger.exception("Disk write completion is unknown; disabling disk I/O and retaining its pages")
                return

            # 释放剩余需要写入的页面
            if remaining_indexes:
                self._release_pages(remaining_indexes)
        else:
            self._release_pages(page_indexes)

    def query_loadable_pages(self, hashs: List[int], start_pos: int) -> int:
        """
        查询从start_pos位置开始,可以从disk cache加载的最长前缀长度
        Returns:
            loadable_len: 从start_pos开始可以加载的长度
        """
        if getattr(self, "_failed", False) or not hashs or start_pos < 0 or start_pos >= len(hashs):
            return 0

        try:
            query_result = self.service.query(hashs)
        except Exception:
            self._failed = True
            logger.exception("Disk lookup failed; recomputing the request prefix")
            return 0
        n = self.service._n
        start_block = start_pos // n
        try:
            first_false_idx = start_block + query_result[start_block:].index(False)
        except ValueError:
            return len(hashs) - start_pos
        first_missing_pos = first_false_idx * n
        return max(0, first_missing_pos - start_pos)

    # 从磁盘读取数据到内存
    def load_pages(self, hashs: List[int], page_indexes: List[int], start_pos: int = 0) -> bool:
        if getattr(self, "_failed", False):
            return False
        if not hashs or not page_indexes or len(hashs) != len(page_indexes):
            return False
        if start_pos < 0 or start_pos >= len(hashs):
            return False

        # 检测当前是否有写操作在进行，若有则跳过本次load请求，暂时不用
        # if self.service.active_threads("w") > 0:
        #     logger.warning("disk cache worker is busy writing, skip load_pages")
        #     return False

        kv_indexer = torch.tensor(page_indexes, dtype=torch.int32, device="cpu")
        task = self._submit_task(hash_128s=hashs, kv_page_indexer=kv_indexer, mode="r", start_pos=start_pos)
        if not self._wait_task(task, "ready"):
            return False
        try:
            return all(state == PyState.Finished for state in task.state())
        except Exception:

            self._failed = True
            logger.exception("Disk read finished without a readable result; recomputing the request prefix")
            return False

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, runtime_checkable

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.log_utils import init_logger

from .tree import Tree


logger = init_logger(__name__)


@dataclass(slots=True)
class CacheAwareConfig:
    cache_threshold: float = 0.5
    balance_rel_threshold: float = 1.2
    eviction_interval_secs: int = 30
    max_tree_size: int = 1000000


class CacheAwarePolicy:
    def __init__(self, config: Optional[CacheAwareConfig] = None) -> None:
        self.config = config or CacheAwareConfig()
        self.tree: Tree = Tree()
        self._stop_eviction = threading.Event()
        self._eviction_thread: Optional[threading.Thread] = None
        if self.config.eviction_interval_secs > 0:
            self._eviction_thread = threading.Thread(
                target=self._run_eviction_loop, name="cache-aware-eviction", daemon=True
            )
            self._eviction_thread.start()

    def _run_eviction_loop(self) -> None:
        while not self._stop_eviction.wait(self.config.eviction_interval_secs):
            logger.info("Running cache eviction...")
            self.evict_cache(self.config.max_tree_size)
            logger.info(f"Cache eviction completed.: {self.tree.get_used_size_per_tenant()}")

    def close(self) -> None:
        self._stop_eviction.set()
        if self._eviction_thread is not None and self._eviction_thread.is_alive():
            self._eviction_thread.join(timeout=1.0)

    def init_workers(self, workers: List[PD_Client_Obj]) -> None:
        for worker in workers:
            self.tree.insert("", worker.url())

    def add_worker(self, worker: PD_Client_Obj) -> None:
        self.tree.insert("", worker.url())

    def remove_worker(self, worker: PD_Client_Obj) -> None:
        self.tree.remove_tenant(worker.url())

    def remove_worker_by_url(self, url: str) -> None:
        self.tree.remove_tenant(url)

    def evict_cache(self, max_size: int) -> None:
        self.tree.evict_tenant_by_size(max_size)

    def _select_worker_min_load(
        self,
        workers: List[PD_Client_Obj],
        request_text: Optional[str],
    ) -> Optional[PD_Client_Obj]:

        min_load_worker = min(workers, key=lambda worker: worker.load())

        if request_text is not None:
            self.tree.insert(request_text, min_load_worker.url())

        return min_load_worker

    def select_worker(
        self, workers: List[PD_Client_Obj], request_text: Optional[str] = None
    ) -> Optional[PD_Client_Obj]:

        if not workers:
            return None

        loads = [worker.load() for worker in workers]
        min_load = min(loads) if loads else 0
        max_load = max(loads) if loads else 0

        is_imbalanced = max_load > (min_load * self.config.balance_rel_threshold)

        logger.info(
            f"CacheAwarePolicy: min_load={min_load:.4f}, max_load={max_load:.4f}, "
            f"balance_rel_threshold={self.config.balance_rel_threshold:.4f}, "
            f"is_imbalanced={is_imbalanced}"
        )

        if is_imbalanced:
            return self._select_worker_min_load(
                workers=workers,
                request_text=request_text,
            )

        text = request_text or ""

        result = self.tree.prefix_match_with_counts(text)
        match_rate = 0.0 if result.input_char_count == 0 else result.matched_char_count / result.input_char_count

        logger.info(
            f"CacheAwarePolicy: matched_char_count={result.matched_char_count}, "
            f"input_char_count={result.input_char_count}, match_rate={match_rate:.4f}, "
            f"cache_threshold={self.config.cache_threshold:.4f}"
        )

        selected_worker: Optional[PD_Client_Obj] = None
        if match_rate > self.config.cache_threshold:
            for worker in workers:
                if worker.url() == result.tenant:
                    selected_worker = worker
                    break

            if selected_worker is None:
                # If the matched tenant is not in the current workers, we can evict it from the tree
                logger.info(f"Evicting tenant: {result.tenant}")
                self.tree.remove_tenant(result.tenant)

        logger.info(
            f"CacheAwarePolicy: selected_worker={selected_worker.url() if selected_worker else None}, "
            f"match_rate={match_rate:.4f}, cache_threshold={self.config.cache_threshold:.4f}"
        )

        if selected_worker is not None:
            self.tree.insert(text, selected_worker.url())
            return selected_worker
        else:
            return self._select_worker_min_load(
                workers=workers,
                request_text=request_text,
            )

    def __del__(self) -> None:
        self.close()

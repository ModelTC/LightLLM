"""
PD Master 的 cache-aware prefill 选点策略。

目标：
  在多 prefill 节点场景下，尽量把 prompt 前缀相近的请求打到同一 P 节点，
  以提高该节点上的前缀 KV cache 命中率；同时在节点负载差距过大时优先做
  负载均衡，避免热点。

实现要点：
  - 用前缀树（见 tree.Tree）记录「历史 prompt -> 处理它的 worker」；
  - 树中的 tenant 对应 worker.client_ip_port；
  - prompt 会按 sample_stride 抽稀后再插入/匹配，降低树的深度与内存；
  - 用 worker.dispatched_prompt_chars（累计派发的 prompt 字符数）做粗粒度均衡。

选点流程见 CacheAwarePolicy.select_worker。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.log_utils import init_logger

from .tree import DEFAULT_SAMPLE_STRIDE, Tree


logger = init_logger(__name__)


@dataclass(slots=True)
class CacheAwareConfig:
    """cache-aware 策略超参。"""

    # 前缀匹配成功率阈值：matched_key_len / input_key_len 超过该值才路由到命中节点。
    cache_threshold: float = 0.5
    # 派发量不均衡判定：max > min * balance_rel_threshold 时强制选派发量最少的节点。
    balance_rel_threshold: float = 1.2
    # 后台按间隔驱逐前缀树中过大的 tenant 缓存；<=0 表示不启动驱逐线程。
    eviction_interval_secs: int = 30
    # 单个 tenant 在前缀树中允许占用的最大 key 字符量（抽稀后的长度口径）。
    max_tree_size: int = 1000000
    # 每隔 sample_stride 个字符抽 1 个作为前缀树 key，降低匹配开销与内存。
    sample_stride: int = DEFAULT_SAMPLE_STRIDE


class CacheAwarePolicy:
    """
    维护 prompt 前缀树，并据此为请求选择 prefill worker。

    树生命周期：
      - 选中 worker 后会把当前 prompt 插入该 worker 对应的 tenant；
      - worker 下线时可通过 remove_worker 清理其 tenant；
      - 后台线程周期性按 max_tree_size 做 LRU 风格驱逐，防止树无限增长。
    """

    def __init__(self, config: Optional[CacheAwareConfig] = None) -> None:
        self.config = config or CacheAwareConfig()
        self.tree: Tree = Tree(sample_stride=self.config.sample_stride)
        self._stop_eviction = threading.Event()
        self._eviction_thread: Optional[threading.Thread] = None
        if self.config.eviction_interval_secs > 0:
            self._eviction_thread = threading.Thread(
                target=self._run_eviction_loop, name="cache-aware-eviction", daemon=True
            )
            self._eviction_thread.start()

    def _run_eviction_loop(self) -> None:
        """周期性裁剪前缀树，控制每个 tenant 的缓存规模。"""
        while not self._stop_eviction.wait(self.config.eviction_interval_secs):
            logger.info("Running cache eviction...")
            self.evict_cache(self.config.max_tree_size)
            logger.info(f"Cache eviction completed.: {self.tree.get_used_size_per_tenant()}")

    def close(self) -> None:
        """停止后台驱逐线程。"""
        self._stop_eviction.set()
        if self._eviction_thread is not None and self._eviction_thread.is_alive():
            self._eviction_thread.join(timeout=1.0)

    def init_workers(self, workers: List[PD_Client_Obj]) -> None:
        """在树中注册一批 worker（插入空前缀，仅建立 tenant）。"""
        for worker in workers:
            self.tree.insert("", worker.client_ip_port)

    def add_worker(self, worker: PD_Client_Obj) -> None:
        """注册单个新上线的 worker。"""
        self.tree.insert("", worker.client_ip_port)

    def remove_worker(self, worker: PD_Client_Obj) -> None:
        """移除下线 worker 在前缀树中的全部记录。"""
        self.tree.remove_tenant(worker.client_ip_port)

    def remove_worker_by_url(self, url: str) -> None:
        """按 client_ip_port 移除 worker 对应 tenant。"""
        self.tree.remove_tenant(url)

    def evict_cache(self, max_size: int) -> None:
        """将各 tenant 占用压缩到 max_size 以下。"""
        self.tree.evict_tenant_by_size(max_size)

    def _select_worker_min_dispatched(
        self,
        workers: List[PD_Client_Obj],
        request_text: Optional[str],
    ) -> Optional[PD_Client_Obj]:
        """
        派发量优先兜底：选择累计 dispatched_prompt_chars 最小的 worker。
        若提供了 request_text，同时把该 prompt 记到该 worker 的前缀树下，
        便于后续相似请求继续命中。
        """
        min_dispatched_worker = min(workers, key=lambda worker: worker.dispatched_prompt_chars)

        if request_text is not None:
            self.tree.insert(request_text, min_dispatched_worker.client_ip_port)

        return min_dispatched_worker

    def select_worker(
        self, workers: List[PD_Client_Obj], request_text: Optional[str] = None
    ) -> Optional[PD_Client_Obj]:
        """
        为一次请求选择 prefill worker。

        决策顺序：
          1) workers 为空 -> 返回 None；
          2) 若 max(dispatched) > min(dispatched) * balance_rel_threshold，
             认为派发不均衡，直接选派发量最少的节点；
          3) 否则对 request_text 做前缀匹配，计算
             match_rate = matched_key_len / input_key_len；
          4) match_rate > cache_threshold 且命中 tenant 仍在线 -> 路由到该节点，
             并更新树；若 tenant 已不在当前 workers 中，则从树中剔除该 tenant；
          5) 未命中阈值或 tenant 失效 -> 回退到派发量最少选择。
        """
        if not workers:
            return None

        # ---- 1. 派发均衡门闩：差距过大时不再追求 cache 亲和 ----
        dispatched_chars = [worker.dispatched_prompt_chars for worker in workers]
        min_dispatched = min(dispatched_chars) if dispatched_chars else 0
        max_dispatched = max(dispatched_chars) if dispatched_chars else 0

        is_imbalanced = max_dispatched > (min_dispatched * self.config.balance_rel_threshold)

        logger.info(
            f"CacheAwarePolicy: min_dispatched={min_dispatched}, max_dispatched={max_dispatched}, "
            f"balance_rel_threshold={self.config.balance_rel_threshold:.4f}, "
            f"is_imbalanced={is_imbalanced}"
        )

        if is_imbalanced:
            return self._select_worker_min_dispatched(
                workers=workers,
                request_text=request_text,
            )

        # ---- 2. 前缀匹配：估计当前请求与历史请求的 cache 复用潜力 ----
        text = request_text or ""

        result = self.tree.prefix_match_with_counts(text)
        # matched/input 均基于抽稀后的 key 长度，比值近似原始前缀重合比例。
        match_rate = 0.0 if result.input_char_count == 0 else result.matched_char_count / result.input_char_count

        logger.info(
            f"CacheAwarePolicy: matched_char_count={result.matched_char_count}, "
            f"input_char_count={result.input_char_count}, match_rate={match_rate:.4f}, "
            f"cache_threshold={self.config.cache_threshold:.4f}"
        )

        selected_worker: Optional[PD_Client_Obj] = None
        if match_rate > self.config.cache_threshold:
            # 树中的 tenant 是 client_ip_port，需要映射回当前在线 worker 对象。
            for worker in workers:
                if worker.client_ip_port == result.tenant:
                    selected_worker = worker
                    break

            if selected_worker is None:
                # 命中了已下线/不在列表中的 tenant，清理脏数据后走派发量兜底。
                logger.info(f"Evicting tenant: {result.tenant}")
                self.tree.remove_tenant(result.tenant)

        logger.info(
            f"CacheAwarePolicy: selected_worker="
            f"{selected_worker.client_ip_port if selected_worker else None}, "
            f"match_rate={match_rate:.4f}, cache_threshold={self.config.cache_threshold:.4f}"
        )

        # ---- 3. 命中则更新树；未命中则派发量兜底并写入树 ----
        if selected_worker is not None:
            self.tree.insert(text, selected_worker.client_ip_port)
            return selected_worker
        else:
            return self._select_worker_min_dispatched(
                workers=workers,
                request_text=request_text,
            )

    def __del__(self) -> None:
        self.close()

"""
PD Master 的 cache-aware prefill 选点策略。

目标：
  在多 prefill 节点场景下，尽量把 prompt 前缀相近的请求打到同一 P 节点，
  以提高该节点上的前缀 KV cache 命中率；同时在节点负载差距过大时优先做
  负载均衡，避免热点。

实现要点：
  - 用前缀树（见 PromptCacheTree）记录「成功进入推理的 prompt -> 处理它的 worker」；
  - 树中的 prefill_node 对应 worker.connection_key，可区分同一服务的不同 DP 连接；
  - prompt 会按 sample_stride 抽稀后再插入/匹配，降低树的深度与内存；
  - 根据推理侧返回的平均 prompt cache 命中率，动态调整 cache 亲和与负载均衡的权重；
  - 以预计未缓存字符数估算新增 Prefill 工作量，缓存节点与冷节点的新增成本不同；
  - 长且高匹配的前缀保持原 P/DP 亲和，避免空闲优先和相对负载阈值反复搬移长请求；
  - dispatched_prompt_chars 记录尚未完成 Prefill 的预计未缓存字符数，而非完整 prompt 长度。

选点流程见 CacheAwarePolicy.select_worker。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.log_utils import init_logger

from .pd_selector import PDSelectionExtraInfo
from .prompt_cache_tree import PromptCacheMatchResult


logger = init_logger(__name__)


@dataclass(slots=True)
class CacheAwareConfig:
    """cache-aware 策略超参。"""

    # 前缀匹配成功率阈值：matched_char_count / input_char_count 超过该值才路由到命中节点。
    cache_threshold: float = 0.5
    # cache 命中节点的在途量超过最空闲节点该倍数时，优先选择最空闲节点。
    balance_rel_threshold: float = 1.5
    # 动态调整 balance_rel_threshold 时允许的上下限。
    min_balance_rel_threshold: float = 1.0
    max_balance_rel_threshold: float = 2.0
    # 每轮用于统计平均 prompt cache 命中率的请求数。
    cache_hit_rate_window_size: int = 1000
    # 相邻统计周期命中率变化时，负载均衡阈值的调整步长。
    balance_rel_threshold_step: float = 0.05
    # 前缀树允许的最大节点数（不含 root）。
    max_node_count: int = 1_000_000
    # 每次 LRU 驱逐的叶节点数量。
    evict_node_batch: int = 10_000
    # 每隔 sample_stride 个字符抽 1 个作为前缀树 key，降低匹配开销与内存。
    sample_stride: int = 512
    # 保护已处理的长前缀；这是历史位置亲和，不代表 KV 一定仍驻留。
    long_prefix_min_chars: int = 65536
    long_prefix_min_hit_rate: float = 0.8


class BalanceRelThresholdController:
    """根据最近请求的 prompt cache 命中率动态调整负载均衡阈值。"""

    def __init__(self) -> None:
        self._cache_hit_rates = []
        self._last_average_cache_hit_rate = None

    def append(self, cache_hit_rate: float) -> None:
        """追加一次真实 prompt cache 命中率。"""
        cache_hit_rate = min(max(cache_hit_rate, 0.0), 1.0)
        self._cache_hit_rates.append(cache_hit_rate)

    def update_config(self, config: CacheAwareConfig) -> None:
        """每收集一个统计窗口，根据命中率趋势调整负载均衡阈值。"""
        if len(self._cache_hit_rates) < config.cache_hit_rate_window_size:
            return

        average_cache_hit_rate = (
            sum(self._cache_hit_rates[-config.cache_hit_rate_window_size :]) / config.cache_hit_rate_window_size
        )
        self._cache_hit_rates.clear()

        if self._last_average_cache_hit_rate is not None:
            if average_cache_hit_rate > self._last_average_cache_hit_rate:
                config.balance_rel_threshold += config.balance_rel_threshold_step
            elif average_cache_hit_rate < self._last_average_cache_hit_rate:
                config.balance_rel_threshold -= config.balance_rel_threshold_step
            config.balance_rel_threshold = min(
                max(config.balance_rel_threshold, config.min_balance_rel_threshold),
                config.max_balance_rel_threshold,
            )

        self._last_average_cache_hit_rate = average_cache_hit_rate


class CacheAwarePolicy:
    """
    使用外部提供的 prompt 前缀树，为请求选择 prefill worker。

    树生命周期：
      - 请求成功进入推理后，把当前 prompt 插入实际 worker 对应的 prefill_node；
      - insert 时若超 max_node_count 会 lazy 触发 LRU 驱逐。
    """

    def __init__(
        self,
        prompt_cache_tree,
        config: CacheAwareConfig,
    ) -> None:
        """初始化选点策略，前缀树和配置均由调用者提供。"""
        self.config = config
        self.prompt_cache_tree = prompt_cache_tree
        self.balance_rel_threshold_controller = BalanceRelThresholdController()

    async def select_worker(
        self, workers: List[PD_Client_Obj], request_text: str, loads=None, fallback_worker=None
    ) -> tuple[Optional[PD_Client_Obj], PDSelectionExtraInfo]:
        """
        为一次请求选择 prefill worker。

        Args:
            workers: 当前可用的 prefill worker 列表。
            request_text: 原始 prompt 文本，长度必须大于 1。

        Raises:
            ValueError: request_text 长度不大于 1 时抛出。

        长且高匹配的前缀保持在线缓存节点亲和；其余命中比较「已有预计工作量 +
        本次未缓存工作量」。只有无有效命中时才优先使用空闲节点。
        """
        if not workers:
            return None, PDSelectionExtraInfo()
        if len(request_text) <= 1:
            raise ValueError(f"request_text length must be > 1, got {len(request_text)}")

        if loads is None:
            loads = {
                worker.connection_key: (worker.dispatched_req_num, worker.dispatched_prompt_chars) for worker in workers
            }
        cache_worker, cache_match = await self._get_cache_worker(workers, request_text)

        request_chars = len(request_text)
        matched_chars = min(request_chars, max(0, cache_match.matched_char_count))
        if (
            cache_worker is not None
            and matched_chars >= self.config.long_prefix_min_chars
            and matched_chars / request_chars >= self.config.long_prefix_min_hit_rate
        ):
            selected_worker = cache_worker
            reason = "long_prefix_affinity"
        elif fallback_worker is not None:
            # Unknown/stale cluster counters are not idle. Preserve long-prefix
            # affinity above, otherwise retain support_ds4's GPU-reported fallback.
            selected_worker = fallback_worker
            reason = "worker_reported_load"
        elif cache_worker is None:
            idle_worker = self._select_idle_worker(workers, loads)
            selected_worker = idle_worker or min(
                workers, key=lambda worker: (loads[worker.connection_key][1], loads[worker.connection_key][0])
            )
            reason = "cold_idle" if idle_worker is not None else "cold_load"
        else:
            selected_worker = self._select_worker_by_cache_and_load(
                workers, cache_worker, request_chars, matched_chars, loads
            )
            reason = "cache_work" if selected_worker is cache_worker else "cold_lower_work"

        selected_cached_chars = matched_chars if selected_worker is cache_worker else 0
        selection_extra_info = PDSelectionExtraInfo(
            estimated_cache_hit_rate=selected_cached_chars / request_chars,
            cache_last_insert_time=cache_match.last_insert_time if selected_worker is cache_worker else None,
            estimated_prefill_chars=max(1, request_chars - selected_cached_chars),
            selection_reason=reason,
            cache_candidate=cache_match.prefill_node,
            cache_candidate_online=cache_worker is not None,
            matched_chars=matched_chars,
            selected_pending_chars=loads[selected_worker.connection_key][1],
        )
        return selected_worker, selection_extra_info

    @staticmethod
    def _matches_worker(result, worker: PD_Client_Obj) -> bool:
        return result.prefill_node == worker.connection_key and (
            result.prefill_node_epoch is None or result.prefill_node_epoch == worker.node_id
        )

    def record_prompt_cache_hit_rate(self, cache_hit_rate: float) -> None:
        """记录推理侧上报的真实 cache 命中率，并更新动态负载阈值。"""
        self.balance_rel_threshold_controller.append(cache_hit_rate)
        self.balance_rel_threshold_controller.update_config(self.config)

    async def _get_cache_worker(
        self, workers: List[PD_Client_Obj], request_text: str, loads=None, fallback_worker=None
    ) -> tuple[Optional[PD_Client_Obj], PromptCacheMatchResult]:
        """在指定候选节点中返回达到匹配阈值的 cache 节点。"""
        result = await self.prompt_cache_tree.prefix_match_sampled(
            request_text[:: self.config.sample_stride],
            len(request_text),
            self.config.sample_stride,
        )
        match_rate = 0.0 if result.input_char_count == 0 else result.matched_char_count / result.input_char_count

        logger.debug(
            f"CacheAwarePolicy: matched_char_count={result.matched_char_count}, "
            f"input_char_count={result.input_char_count}, match_rate={match_rate:.4f}, "
            f"cache_threshold={self.config.cache_threshold:.4f}, "
            f"prefill_node={result.prefill_node}"
        )

        if match_rate <= self.config.cache_threshold or result.prefill_node is None:
            return None, result

        for worker in workers:
            if self._matches_worker(result, worker):
                return worker, result
        return None, result

    def _select_idle_worker(self, workers: List[PD_Client_Obj], loads) -> Optional[PD_Client_Obj]:
        """随机返回一个空闲节点。"""
        idle_workers = [worker for worker in workers if loads[worker.connection_key][0] == 0]
        if not idle_workers:
            return None

        selected_worker = random.choice(idle_workers)
        logger.debug(
            f"CacheAwarePolicy: select idle worker, idle_worker_num={len(idle_workers)}, "
            f"balance_rel_threshold={self.config.balance_rel_threshold:.4f}, "
            f"selected_worker={selected_worker.client_ip_port}"
        )
        return selected_worker

    def _select_worker_by_cache_and_load(
        self,
        workers: List[PD_Client_Obj],
        cache_worker: Optional[PD_Client_Obj],
        request_load: int,
        cached_chars: int = 0,
        loads=None,
    ) -> PD_Client_Obj:
        """所有节点都忙时，在 cache 亲和与 prompt 负载之间选择节点。"""
        if loads is None:
            loads = {
                worker.connection_key: (worker.dispatched_req_num, worker.dispatched_prompt_chars) for worker in workers
            }
        least_loaded_worker = min(
            workers, key=lambda worker: (loads[worker.connection_key][1], loads[worker.connection_key][0])
        )
        least_projected_load = loads[least_loaded_worker.connection_key][1] + request_load
        cache_projected_load = None
        cache_worker_is_overloaded = False
        if cache_worker is not None:
            cache_projected_load = loads[cache_worker.connection_key][1] + max(1, request_load - cached_chars)
            cache_worker_is_overloaded = cache_projected_load > least_projected_load * self.config.balance_rel_threshold

        if cache_worker is None or cache_worker_is_overloaded:
            selected_worker = least_loaded_worker
        else:
            selected_worker = cache_worker

        logger.debug(
            f"CacheAwarePolicy: cache_worker={cache_worker.client_ip_port if cache_worker else None}, "
            f"cache_worker_load={loads[cache_worker.connection_key][1] if cache_worker else None}, "
            f"cache_projected_load={cache_projected_load}, "
            f"least_loaded_worker={least_loaded_worker.client_ip_port}, "
            f"least_loaded_worker_load={loads[least_loaded_worker.connection_key][1]}, "
            f"least_projected_load={least_projected_load}, "
            f"balance_rel_threshold={self.config.balance_rel_threshold:.4f}, "
            f"cache_worker_is_overloaded={cache_worker_is_overloaded}, "
            f"selected_worker={selected_worker.client_ip_port}"
        )
        return selected_worker

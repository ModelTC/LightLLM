"""
PD Master 的 cache-aware prefill 选点策略。

目标：
  在多 prefill 节点场景下，尽量把 prompt 前缀相近的请求打到同一 P 节点，
  以提高该节点上的前缀 KV cache 命中率；同时在节点负载差距过大时优先做
  负载均衡，避免热点。

实现要点：
  - 用前缀树（见 PromptCacheTree）记录「成功进入推理的 prompt -> 处理它的 worker」；
  - 树中的 prefill_node 对应 worker.client_ip_port；
  - prompt 会按 sample_stride 抽稀后再插入/匹配，降低树的深度与内存；
  - 根据推理侧返回的平均 prompt cache 命中率，动态调整 cache 亲和与负载均衡的权重；
  - 优先使用 dispatched_req_num 为 0 的空闲节点，避免 GPU 闲置；
  - 所有节点都忙时，用 worker.dispatched_prompt_chars 做负载均衡。

选点流程见 CacheAwarePolicy.select_worker。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.log_utils import init_logger

from .prompt_cache_tree import PromptCacheTree


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
    # 初始化前缀树时通过 sys.setrecursionlimit 调大 Python 调用栈深度。
    recursion_limit: int = 4000


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
    维护 prompt 前缀树，并据此为请求选择 prefill worker。

    树生命周期：
      - 请求成功进入推理后，把当前 prompt 插入实际 worker 对应的 prefill_node；
      - insert 时若超 max_node_count 会 lazy 触发 LRU 驱逐。
    """

    def __init__(self, config: Optional[CacheAwareConfig] = None) -> None:
        """初始化前缀树。"""
        self.config = config or CacheAwareConfig()
        self.prompt_cache_tree: PromptCacheTree = PromptCacheTree(
            sample_stride=self.config.sample_stride,
            max_node_count=self.config.max_node_count,
            evict_node_batch=self.config.evict_node_batch,
            recursion_limit=self.config.recursion_limit,
        )
        self.balance_rel_threshold_controller = BalanceRelThresholdController()

    def select_worker(self, workers: List[PD_Client_Obj], request_text: str) -> Optional[PD_Client_Obj]:
        """
        为一次请求选择 prefill worker。

        Args:
            workers: 当前可用的 prefill worker 列表。
            request_text: 原始 prompt 文本，长度必须大于 1。

        Raises:
            ValueError: request_text 长度不大于 1 时抛出。

        决策顺序：
          1) workers 为空 -> 返回 None；
          2) 存在 dispatched_req_num 为 0 的空闲节点 -> 强制从空闲节点中选择；
             多个节点空闲时优先选择 cache 命中节点；
          3) 对 request_text 做前缀匹配，计算
             match_rate = matched_char_count / input_char_count；
          4) match_rate > cache_threshold 且命中 prefill_node 仍在线 -> 得到 cache 命中节点；
          5) cache 命中节点负载未严重高于最空闲节点 -> 选择 cache 命中节点；
          6) 未命中或负载严重失衡 -> 选择最空闲节点；
        """
        if not workers:
            return None
        if len(request_text) <= 1:
            raise ValueError(f"request_text length must be > 1, got {len(request_text)}")

        # ---- 1. 空闲优先：避免有可用 GPU 闲置 ----
        idle_worker = self._select_idle_worker(workers, request_text)
        if idle_worker is not None:
            return idle_worker

        # ---- 2. 所有节点都忙时，在 cache 亲和与负载均衡之间权衡 ----
        cache_worker = self._get_cache_worker(workers, request_text)
        selected_worker = self._select_worker_by_cache_and_load(workers, cache_worker, len(request_text))
        return selected_worker

    def get_estimated_cache_hit_rate(self, selected_worker: PD_Client_Obj, request_text: str) -> float:
        """查询最终选中节点的输入 cache 命中率估计。"""
        result = self.prompt_cache_tree.prefix_match(request_text)
        if result.prefill_node != selected_worker.client_ip_port or result.input_char_count == 0:
            return 0.0
        return result.matched_char_count / result.input_char_count

    def insert_prompt_cache(self, request_text: str, selected_worker: PD_Client_Obj) -> None:
        """在请求成功进入推理后，记录 prompt 与实际执行的 Prefill 节点。"""
        self.prompt_cache_tree.insert(request_text, selected_worker.client_ip_port)

    def record_prompt_cache_hit_rate(self, cache_hit_rate: float) -> None:
        """记录推理侧上报的真实 cache 命中率，并更新动态负载阈值。"""
        self.balance_rel_threshold_controller.append(cache_hit_rate)
        self.balance_rel_threshold_controller.update_config(self.config)

    def _get_cache_worker(self, workers: List[PD_Client_Obj], request_text: str) -> Optional[PD_Client_Obj]:
        """在指定候选节点中返回达到匹配阈值的 cache 节点。"""
        result = self.prompt_cache_tree.prefix_match(request_text)
        match_rate = 0.0 if result.input_char_count == 0 else result.matched_char_count / result.input_char_count

        logger.info(
            f"CacheAwarePolicy: matched_char_count={result.matched_char_count}, "
            f"input_char_count={result.input_char_count}, match_rate={match_rate:.4f}, "
            f"cache_threshold={self.config.cache_threshold:.4f}, "
            f"prefill_node={result.prefill_node}"
        )

        if match_rate <= self.config.cache_threshold or result.prefill_node is None:
            return None

        for worker in workers:
            if worker.client_ip_port == result.prefill_node:
                return worker
        return None

    def _select_idle_worker(self, workers: List[PD_Client_Obj], request_text: str) -> Optional[PD_Client_Obj]:
        """优先选择空闲节点；多个空闲节点之间优先复用 cache。"""
        idle_workers = [worker for worker in workers if worker.dispatched_req_num == 0]
        if not idle_workers:
            return None

        cache_worker = self._get_cache_worker(idle_workers, request_text) if len(idle_workers) > 1 else None
        selected_worker = cache_worker or min(
            idle_workers,
            key=lambda worker: (worker.dispatched_prompt_chars, worker.client_ip_port),
        )
        logger.info(
            f"CacheAwarePolicy: select idle worker, idle_worker_num={len(idle_workers)}, "
            f"cache_worker={cache_worker.client_ip_port if cache_worker else None}, "
            f"balance_rel_threshold={self.config.balance_rel_threshold:.4f}, "
            f"selected_worker={selected_worker.client_ip_port}"
        )
        return selected_worker

    def _select_worker_by_cache_and_load(
        self,
        workers: List[PD_Client_Obj],
        cache_worker: Optional[PD_Client_Obj],
        request_load: int,
    ) -> PD_Client_Obj:
        """所有节点都忙时，在 cache 亲和与 prompt 负载之间选择节点。"""
        least_loaded_worker = min(workers, key=lambda worker: worker.dispatched_prompt_chars)
        least_projected_load = least_loaded_worker.dispatched_prompt_chars + request_load
        cache_projected_load = None
        cache_worker_is_overloaded = False
        if cache_worker is not None:
            cache_projected_load = cache_worker.dispatched_prompt_chars + request_load
            cache_worker_is_overloaded = cache_projected_load > least_projected_load * self.config.balance_rel_threshold

        if cache_worker is None or cache_worker_is_overloaded:
            selected_worker = least_loaded_worker
        else:
            selected_worker = cache_worker

        logger.info(
            f"CacheAwarePolicy: cache_worker={cache_worker.client_ip_port if cache_worker else None}, "
            f"cache_worker_load={cache_worker.dispatched_prompt_chars if cache_worker else None}, "
            f"cache_projected_load={cache_projected_load}, "
            f"least_loaded_worker={least_loaded_worker.client_ip_port}, "
            f"least_loaded_worker_load={least_loaded_worker.dispatched_prompt_chars}, "
            f"least_projected_load={least_projected_load}, "
            f"balance_rel_threshold={self.config.balance_rel_threshold:.4f}, "
            f"cache_worker_is_overloaded={cache_worker_is_overloaded}, "
            f"selected_worker={selected_worker.client_ip_port}"
        )
        return selected_worker

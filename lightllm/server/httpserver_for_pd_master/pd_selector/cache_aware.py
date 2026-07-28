"""
PD Master 的 cache-aware prefill 选点策略。

目标：
  在多 prefill 节点场景下，尽量把 prompt 前缀相近的请求打到同一 P 节点，
  以提高该节点上的前缀 KV cache 命中率；同时在节点负载差距过大时优先做
  负载均衡，避免热点。

实现要点：
  - 用前缀树（见 PromptCacheTree）记录「历史 prompt -> 处理它的 worker」；
  - 树中的 prefill_node 对应 worker.client_ip_port；
  - prompt 会按 sample_stride 抽稀后再插入/匹配，降低树的深度与内存；
  - 用 worker.dispatched_prompt_chars（累计派发的 prompt 字符数）做粗粒度均衡。

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
    # 派发量不均衡判定：max > min * balance_rel_threshold 时强制选派发量最少的节点。
    balance_rel_threshold: float = 1.2
    # 前缀树允许的最大节点数（不含 root）。
    max_node_count: int = 1_000_000
    # 每次 LRU 驱逐的叶节点数量。
    evict_node_batch: int = 10_000
    # 每隔 sample_stride 个字符抽 1 个作为前缀树 key，降低匹配开销与内存。
    sample_stride: int = 512
    # 初始化前缀树时通过 sys.setrecursionlimit 调大 Python 调用栈深度。
    recursion_limit: int = 4000


class CacheAwarePolicy:
    """
    维护 prompt 前缀树，并据此为请求选择 prefill worker。

    树生命周期：
      - 选中 worker 后会把当前 prompt 插入该 worker 对应的 prefill_node；
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

    def _select_worker_min_dispatched(
        self,
        workers: List[PD_Client_Obj],
        request_text: str,
    ) -> PD_Client_Obj:
        """派发量优先兜底：选择累计 dispatched_prompt_chars 最小的 worker，并写入前缀树。"""
        min_dispatched_worker = min(workers, key=lambda worker: worker.dispatched_prompt_chars)
        self.prompt_cache_tree.insert(request_text, min_dispatched_worker.client_ip_port)
        return min_dispatched_worker

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
          2) 若 max(dispatched) > min(dispatched) * balance_rel_threshold，
             认为派发不均衡，直接选派发量最少的节点；
          3) 否则对 request_text 做前缀匹配，计算
             match_rate = matched_char_count / input_char_count；
          4) match_rate > cache_threshold 且命中 prefill_node 仍在线 -> 路由到该节点并更新树；
          5) 未命中阈值或 prefill_node 不在当前 workers 中 -> 回退到派发量最少选择。
        """
        if not workers:
            return None
        if len(request_text) <= 1:
            raise ValueError(f"request_text length must be > 1, got {len(request_text)}")

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
        result = self.prompt_cache_tree.prefix_match(request_text)
        match_rate = 0.0 if result.input_char_count == 0 else result.matched_char_count / result.input_char_count

        logger.info(
            f"CacheAwarePolicy: matched_char_count={result.matched_char_count}, "
            f"input_char_count={result.input_char_count}, match_rate={match_rate:.4f}, "
            f"cache_threshold={self.config.cache_threshold:.4f}, "
            f"prefill_node={result.prefill_node}"
        )

        selected_worker: Optional[PD_Client_Obj] = None
        if match_rate > self.config.cache_threshold and result.prefill_node is not None:
            # 树中的 prefill_node 是 client_ip_port，需要映射回当前在线 worker 对象。
            for worker in workers:
                if worker.client_ip_port == result.prefill_node:
                    selected_worker = worker
                    break

        logger.info(
            f"CacheAwarePolicy: selected_worker="
            f"{selected_worker.client_ip_port if selected_worker else None}, "
            f"match_rate={match_rate:.4f}, cache_threshold={self.config.cache_threshold:.4f}"
        )

        # ---- 3. 命中则更新树；未命中则派发量兜底并写入树 ----
        if selected_worker is not None:
            self.prompt_cache_tree.insert(request_text, selected_worker.client_ip_port)
            return selected_worker
        else:
            return self._select_worker_min_dispatched(
                workers=workers,
                request_text=request_text,
            )

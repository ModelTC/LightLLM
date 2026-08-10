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
  - 用 worker.dispatched_prompt_chars（当前尚未产出首 token 的 prompt 字符数）做负载均衡。

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
    balance_rel_threshold: float = 1.8
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
          2) 对 request_text 做前缀匹配，计算
             match_rate = matched_char_count / input_char_count；
          3) match_rate > cache_threshold 且命中 prefill_node 仍在线 -> 得到 cache 命中节点；
          4) cache 命中节点负载未严重高于最空闲节点 -> 选择 cache 命中节点；
          5) 未命中或负载严重失衡 -> 选择最空闲节点；
          6) 将当前 prompt 与最终选中的节点写入前缀树。
        """
        if not workers:
            return None
        if len(request_text) <= 1:
            raise ValueError(f"request_text length must be > 1, got {len(request_text)}")

        # ---- 1. 前缀匹配：估计当前请求与历史请求的 cache 复用潜力 ----
        result = self.prompt_cache_tree.prefix_match(request_text)
        match_rate = 0.0 if result.input_char_count == 0 else result.matched_char_count / result.input_char_count

        logger.info(
            f"CacheAwarePolicy: matched_char_count={result.matched_char_count}, "
            f"input_char_count={result.input_char_count}, match_rate={match_rate:.4f}, "
            f"cache_threshold={self.config.cache_threshold:.4f}, "
            f"prefill_node={result.prefill_node}"
        )

        cache_worker: Optional[PD_Client_Obj] = None
        if match_rate > self.config.cache_threshold and result.prefill_node is not None:
            for worker in workers:
                if worker.client_ip_port == result.prefill_node:
                    cache_worker = worker
                    break

        # ---- 2. 负载配平：cache 节点过载时选择当前在途量最少的节点 ----
        least_loaded_worker = min(workers, key=lambda worker: worker.dispatched_prompt_chars)
        request_load = len(request_text)
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

        self.prompt_cache_tree.insert(request_text, selected_worker.client_ip_port)
        return selected_worker

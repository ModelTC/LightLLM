import asyncio
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.utils.error_utils import PDTransportUnavailableError, ServerBusyError
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


@dataclass(slots=True)
class PDSelectionExtraInfo:
    """PD 节点调度的附加返回信息。"""

    estimated_cache_hit_rate: float = 0.0
    cache_last_insert_time: Optional[float] = None
    # 字符数是负载估算单位，不是真实 token/KV 命中数；非 cache-aware 策略沿用完整 prompt。
    estimated_prefill_chars: Optional[int] = field(default=None, compare=False)
    selection_reason: str = field(default="default", compare=False)
    cache_candidate: Optional[str] = field(default=None, compare=False)
    cache_candidate_online: bool = field(default=False, compare=False)
    matched_chars: int = field(default=0, compare=False)
    selected_pending_chars: int = field(default=0, compare=False)


class PDSelector:
    def __init__(self, pd_manager):
        from .pd_node_stats_client import create_pd_node_stats_client

        self.prefill_nodes: List[PD_Client_Obj] = []
        self.decode_nodes: List[PD_Client_Obj] = []
        self.pd_manager = pd_manager
        self.pd_node_stats = create_pd_node_stats_client()
        # 只有依赖共享在途统计的策略需要覆盖“读取、选点、预占”全过程。
        # 其他策略使用异步空上下文，避免无意义地串行化请求。
        self.selection_lock = nullcontext()

    def update_nodes(self, prefill_nodes, decode_nodes):
        self.prefill_nodes = [node for node in prefill_nodes if node.run_status.transfer_available]
        self.decode_nodes = [node for node in decode_nodes if node.run_status.transfer_available]

    def _check_available_nodes(self):
        if not self.prefill_nodes or not self.decode_nodes:
            raise PDTransportUnavailableError(
                "PD transport unavailable: "
                f"available_prefill={len(self.prefill_nodes)}, available_decode={len(self.decode_nodes)}"
            )

    async def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj, PDSelectionExtraInfo]:
        raise NotImplementedError("Subclass must implement this method")

    def record_prompt_cache_hit_rate(self, cache_hit_rate: float) -> None:
        """记录推理侧返回的 prompt cache 命中率；非 cache-aware 策略无需处理。"""
        return

    async def insert_prompt_cache(self, prompt: str, p_node: PD_Client_Obj) -> None:
        """记录成功进入推理的 prompt；非 cache-aware 策略无需处理。"""
        return

    async def run_metadata_sync(self) -> None:
        return


class RandomSelector(PDSelector):
    """随机选择器"""

    async def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj, PDSelectionExtraInfo]:
        self._check_available_nodes()
        p_node = random.choice(self.prefill_nodes)
        d_node = random.choice(self.decode_nodes)
        return p_node, d_node, PDSelectionExtraInfo()


class RoundRobinSelector(PDSelector):
    """轮询选择器"""

    def __init__(self, pd_manager):
        super().__init__(pd_manager)
        self.prefill_node_index: int = 0
        self.decode_node_index: int = 0

    async def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj, PDSelectionExtraInfo]:
        self._check_available_nodes()
        self.prefill_node_index = self.prefill_node_index % len(self.prefill_nodes)
        self.decode_node_index = self.decode_node_index % len(self.decode_nodes)
        p_node = self.prefill_nodes[self.prefill_node_index]
        d_node = self.decode_nodes[self.decode_node_index]
        self.prefill_node_index += 1
        self.decode_node_index += 1
        return p_node, d_node, PDSelectionExtraInfo()


class AdaptiveLoadSelector(PDSelector):
    """基于负载使用情况的选择器"""

    async def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj, PDSelectionExtraInfo]:
        self._check_available_nodes()
        p_node = self._importance_sampling(self.prefill_nodes)
        d_node = self._importance_sampling(self.decode_nodes)

        return p_node, d_node, PDSelectionExtraInfo()

    def _importance_sampling(self, nodes: List[PD_Client_Obj]):
        return random.choices(nodes, weights=[max(1.0 - e.run_status.total_token_usage_rate, 0.02) for e in nodes])[0]


class LoadBalancedCacheAwareSelector(AdaptiveLoadSelector):
    """Cache-aware prefill 选点：按抽稀后的 prompt 前缀匹配 + 负载均衡。"""

    def __init__(self, pd_manager):
        from .prompt_cache_tree_client import create_prompt_cache_tree_client
        from .cache_aware import CacheAwareConfig, CacheAwarePolicy

        super().__init__(pd_manager)
        self.selection_lock = asyncio.Lock()
        config = CacheAwareConfig()
        self.policy = CacheAwarePolicy(
            prompt_cache_tree=create_prompt_cache_tree_client(config=config),
            config=config,
        )
        self.shared_cache = None
        if pd_manager is not None and pd_manager.args.config_server_host and pd_manager.args.config_server_port:
            from .shared_cache import SharedCacheState

            self.shared_cache = SharedCacheState(self, pd_manager.args)

    async def select_p_d_node(
        self, prompt: Union[str, List[int]], sampling_params: SamplingParams, multimodal_params: MultimodalParams
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj, PDSelectionExtraInfo]:
        self._check_available_nodes()
        assert isinstance(prompt, str), "prompt must be a string for cache-aware selection"
        if self.shared_cache is None:
            await self.pd_node_stats.update_dispatched_info(p_nodes=self.prefill_nodes)
            p_node, selection_extra_info = await self.policy.select_worker(self.prefill_nodes, request_text=prompt)
        elif not self.shared_cache.enabled():
            return self.prefill_nodes[0], self._importance_sampling(self.decode_nodes), PDSelectionExtraInfo()
        else:
            loads = self.shared_cache.cluster_loads(self.prefill_nodes)
            if loads is None:
                p_node, selection_extra_info = await self.policy.select_worker(
                    self.prefill_nodes, prompt, fallback_worker=self._importance_sampling(self.prefill_nodes)
                )
            else:
                p_node, selection_extra_info = await self.policy.select_worker(self.prefill_nodes, prompt, loads)
        self._check_available_nodes()
        # 前缀查询期间 P 节点可能已经重连；即使 connection_key 相同，旧对象持有的 WebSocket 也已失效。
        # 必须按对象身份确认选中节点仍是当前连接，让上层通过 ServerBusyError 重新选点。
        if not any(node is p_node for node in self.prefill_nodes):
            raise ServerBusyError("Selected PD connection changed during cache lookup")
        d_node = self._importance_sampling(self.decode_nodes)

        logger.debug(
            f"LoadBalancedCacheAwareSelector: selected p_node={p_node.client_ip_port}, "
            f"d_node={d_node.client_ip_port}, "
            f"cache_last_insert_time={selection_extra_info.cache_last_insert_time}"
        )

        return p_node, d_node, selection_extra_info

    def record_prompt_cache_hit_rate(self, cache_hit_rate: float) -> None:
        self.policy.record_prompt_cache_hit_rate(cache_hit_rate)

    async def insert_prompt_cache(self, prompt: str, p_node: PD_Client_Obj) -> None:
        if self.shared_cache is not None:
            self.shared_cache.publish_hint(prompt, p_node)
            return
        sample_stride = self.policy.config.sample_stride
        await self.policy.prompt_cache_tree.insert_sampled(
            prompt[::sample_stride], p_node.connection_key, sample_stride
        )

    async def run_metadata_sync(self) -> None:
        if self.shared_cache is not None:
            await self.shared_cache.run()

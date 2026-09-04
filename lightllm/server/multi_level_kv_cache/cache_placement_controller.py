import enum
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, List, Optional, Protocol, Sequence, Tuple

from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.server.core.objs import StartArgs
    from lightllm.server.router.model_infer.infer_batch import InferReq

logger = init_logger(__name__)


class _RadixCache(Protocol):
    total_token_num: int


class CacheTier(str, enum.Enum):
    """单个缓存层级。"""

    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"


_VALID_CACHE_TIER_TUPLES = {
    (CacheTier.GPU,),
    (CacheTier.CPU,),
    (CacheTier.CPU, CacheTier.DISK),
    (CacheTier.GPU, CacheTier.CPU),
    (CacheTier.GPU, CacheTier.CPU, CacheTier.DISK),
}


@dataclass(frozen=True)
class CacheCapacityConfig:
    """以可容纳的 prompt token 数量表示各级缓存容量。"""

    gpu_tokens: int
    cpu_tokens: int
    disk_tokens: int

    @property
    def total_tokens(self) -> int:
        # Disk 必须经过 CPU，二者是串联关系，低层有效容量取较大值而不是相加。
        return self.gpu_tokens + max(self.cpu_tokens, self.disk_tokens)


class CachePlacementController(ABC):
    """缓存放置策略接口。"""

    @abstractmethod
    def set_req_cache_way(self, reqs: Sequence["InferReq"]) -> None:
        """为尚未分配缓存方式的请求设置 ``cache_tiers``。"""


class GpuOnlyCachePlacementController(CachePlacementController):
    """未启用多级缓存时，仅将请求写入 GPU cache。"""

    def set_req_cache_way(self, reqs: Sequence["InferReq"]) -> None:
        assert all(req.cache_tiers == (CacheTier.GPU,) for req in reqs)


class LegacyCachePlacementController(CachePlacementController):
    """兼容旧行为，将请求同时写入所有已启用的缓存层级。"""

    def __init__(self, enable_cpu_cache: bool, enable_disk_cache: bool):
        if enable_disk_cache:
            self._cache_tiers = (CacheTier.GPU, CacheTier.CPU, CacheTier.DISK)
        elif enable_cpu_cache:
            self._cache_tiers = (CacheTier.GPU, CacheTier.CPU)
        else:
            self._cache_tiers = (CacheTier.GPU,)
        assert self._cache_tiers in _VALID_CACHE_TIER_TUPLES

    def set_req_cache_way(self, reqs: Sequence["InferReq"]) -> None:
        assert all(req.cache_tiers == (CacheTier.GPU,) for req in reqs)
        for req in reqs:
            # Diverse 模式下只有主请求支持向 CPU/Disk cache 卸载，从请求仍只写入 GPU cache。
            if req.shm_req.group_req_id == req.shm_req.request_id:
                req.cache_tiers = self._cache_tiers
            else:
                req.cache_tiers = (CacheTier.GPU,)


class AdaptiveCachePlacementController(CachePlacementController):
    """
    根据近期请求长度和缓存容量，在 GPU 与低层缓存路径之间自适应放置请求。

    控制器先用较小的初始窗口快速生成长度分界点，之后保留最近的请求输入长度，
    每隔固定步数根据累计 token 数量和 GPU 与低层有效容量比例更新分界点。Disk 必须
    通过 CPU 中转，因此低层有效容量取 CPU、Disk 容量的较大值，Disk 目标使用
    ``(CPU, Disk)`` 路径表示。
    """

    INITIAL_HISTORY_SIZE = 128
    MAX_HISTORY_SIZE = 512
    UPDATE_INTERVAL_STEPS = 36

    def __init__(
        self,
        capacity: CacheCapacityConfig,
        args: "StartArgs",
        max_history_size: int = MAX_HISTORY_SIZE,
        initial_history_size: int = INITIAL_HISTORY_SIZE,
    ):
        assert 0 < initial_history_size <= max_history_size
        self._capacity = capacity
        self._args = args
        self._initial_history_size = initial_history_size
        self._recent_input_lengths: Deque[int] = deque(maxlen=max_history_size)
        self._gpu_max_input_len: Optional[int] = None
        self._steps_since_last_update = 0
        self._legacy_controller = LegacyCachePlacementController(
            enable_cpu_cache=args.enable_cpu_cache,
            enable_disk_cache=args.enable_disk_cache,
        )
        self._lock = threading.Lock()

    def set_req_cache_way(self, reqs: Sequence["InferReq"]) -> None:
        with self._lock:
            assert all(req.cache_tiers == (CacheTier.GPU,) for req in reqs)
            pending_master_reqs: List["InferReq"] = []
            pending_slave_reqs: List["InferReq"] = []
            for req in reqs:
                if req.shm_req.group_req_id == req.shm_req.request_id:
                    pending_master_reqs.append(req)
                else:
                    pending_slave_reqs.append(req)

            input_lengths = [req.shm_req.input_len for req in pending_master_reqs]

            # Diverse 模式下的从请求继续使用 GPU radix cache，只有主请求会被记录到历史窗口中。
            for req in pending_slave_reqs:
                req.cache_tiers = (CacheTier.GPU,)

            if not pending_master_reqs:
                return

            if self._gpu_max_input_len is None:
                self._legacy_controller.set_req_cache_way(pending_master_reqs)
            else:
                for req, input_length in zip(pending_master_reqs, input_lengths):
                    cache_tiers = self._select_cache_tiers(input_length=input_length)
                    assert cache_tiers in _VALID_CACHE_TIER_TUPLES
                    req.cache_tiers = cache_tiers

            # 先用小窗口快速生成首个边界；之后保留最近最多 max_history_size 条数据形成滑动窗口。
            # deque 超过 maxlen 时会自动从左侧淘汰旧数据，每累计固定步数重新计算以跟踪负载变化趋势。
            self._recent_input_lengths.extend(input_lengths)
            if self._gpu_max_input_len is None:
                if len(self._recent_input_lengths) >= self._initial_history_size:
                    self._gpu_max_input_len = self._calculate_gpu_max_input_len()
            else:
                self._steps_since_last_update += 1
                if self._steps_since_last_update >= self.UPDATE_INTERVAL_STEPS:
                    self._gpu_max_input_len = self._calculate_gpu_max_input_len()
                    self._steps_since_last_update = 0

    def _calculate_gpu_max_input_len(self) -> int:
        # 没有 GPU radix cache 容量时，不应将任何正常请求分配到 GPU cache。
        if self._capacity.gpu_tokens == 0:
            return 0

        # 按输入长度从小到大排列，使每个相邻长度都可以作为 GPU 与低层缓存的候选分界点。
        sorted_lengths = sorted(self._recent_input_lengths)

        # 使用 token 数量而不是请求数量作为权重，避免大量短请求与少量长请求被等同计算。
        total_weight = sum(max(1, length) for length in sorted_lengths)

        # 按 GPU 容量在 GPU + 低层有效容量中的比例，计算本窗口期望放入 GPU 的 token 数量。
        gpu_target = total_weight * self._capacity.gpu_tokens / self._capacity.total_tokens

        # 顺序累加较短请求的 token 数量，首次超过目标值时直接返回当前输入长度作为分界点。
        prefix_sum = 0
        for length in sorted_lengths:
            prefix_sum += max(1, length)
            if prefix_sum > gpu_target:
                return length

        # 累计值始终没有超过目标时，全部请求都属于 GPU 范围。
        return sorted_lengths[-1]

    def _select_cache_tiers(self, input_length: int) -> Tuple[CacheTier, ...]:
        assert self._gpu_max_input_len is not None
        if input_length <= self._gpu_max_input_len:
            return (CacheTier.GPU,)
        if self._args.enable_disk_cache:
            return (CacheTier.CPU, CacheTier.DISK)
        if self._args.enable_cpu_cache:
            return (CacheTier.CPU,)
        return (CacheTier.GPU,)


def create_cache_placement_controller(
    args: "StartArgs", radix_cache: Optional[_RadixCache]
) -> CachePlacementController:
    """根据服务配置创建对应的缓存放置控制器。"""
    if not args.enable_cpu_cache:
        controller = GpuOnlyCachePlacementController()
    elif args.cache_placement_strategy == "legacy":
        controller = LegacyCachePlacementController(
            enable_cpu_cache=args.enable_cpu_cache,
            enable_disk_cache=args.enable_disk_cache,
        )
    else:
        assert args.cache_placement_strategy == "adaptive"
        from lightllm.utils.envs_utils import get_cache_placement_gpu_capacity_ratio
        from lightllm.utils.kv_cache_utils import calcu_cpu_cache_meta

        cpu_cache_meta = calcu_cpu_cache_meta()
        page_size_tokens = args.cpu_cache_token_page_size
        gpu_capacity_ratio = get_cache_placement_gpu_capacity_ratio()

        if radix_cache is None:
            physical_gpu_capacity_tokens = 0
            gpu_capacity_tokens = 0
        else:
            # CPU/Disk cache 在节点内由所有 DP 实例共享，GPU 容量需按当前节点的 DP 数量汇总。
            dp_size_in_node = max(1, args.dp // args.nnodes)
            physical_gpu_capacity_tokens = radix_cache.total_token_num * dp_size_in_node
            # 为运行态请求预留部分 GPU KV 容量，只用指定比例估算可长期保留的 GPU cache 容量。
            gpu_capacity_tokens = int(physical_gpu_capacity_tokens * gpu_capacity_ratio)

        cpu_capacity_tokens = cpu_cache_meta.page_num * page_size_tokens

        if args.enable_disk_cache:
            disk_capacity_bytes = int(args.disk_cache_storage_size * (1024 ** 3))
            disk_capacity_tokens = disk_capacity_bytes // cpu_cache_meta.calcu_one_page_size() * page_size_tokens
        else:
            disk_capacity_tokens = 0

        logger.info(
            "cache placement capacities in tokens: gpu=%d, physical_gpu=%d, gpu_ratio=%.4f, cpu=%d, disk=%d",
            gpu_capacity_tokens,
            physical_gpu_capacity_tokens,
            gpu_capacity_ratio,
            cpu_capacity_tokens,
            disk_capacity_tokens,
        )
        controller = AdaptiveCachePlacementController(
            capacity=CacheCapacityConfig(
                gpu_tokens=gpu_capacity_tokens,
                cpu_tokens=cpu_capacity_tokens,
                disk_tokens=disk_capacity_tokens,
            ),
            args=args,
        )

    return controller

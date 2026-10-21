from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from sortedcontainers import SortedDict

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


@dataclass(frozen=True)
class SpecDecodePlan:
    """Planner decision for one target decode iteration.

    ``origin_batch_size`` records the physical target row count before planning,
    while ``dynamic_batch_size`` is the row count selected for this target
    forward. Fixed scheduling sets both values to the same size; dynamic
    scheduling may reduce ``dynamic_batch_size`` before the forward. Therefore
    both modes share the same plan representation without using ``None`` as a
    mode marker.

    In addition:
    - draft_step is the candidate length to generate after target verify
    - pre_draft_step describes the previous iteration and controls whether
      GPU verify sync can be skipped
    """

    origin_batch_size: int
    dynamic_batch_size: int
    draft_step: int
    pre_draft_step: int
    # False when the current batch contains requests without a proposal from
    # the previous iteration. Such an iteration does not represent one
    # well-defined LightSpec runtime configuration.
    all_reqs_have_proposals: bool = True

    @property
    def skip_verify_sync(self) -> bool:
        return self.pre_draft_step == 0


class BaseMtpPlanner(ABC):
    """定义 SpecEngine 与不同 MTP 规划器之间的统一调用接口。"""

    @abstractmethod
    def plan(self, decode_reqs: List, origin_batch_size: int) -> SpecDecodePlan:
        """为当前 decode 迭代生成执行计划。

        Args:
            decode_reqs: 当前参与 decode 的逻辑请求列表。规划器可以读取
                请求的输出进度，判断请求是否已经持有上一轮生成的
                draft proposal；DP 空 rank 对应空列表。
            origin_batch_size: 进入动态压缩前的物理 verify 行数。

        Returns:
            本轮 target verify 使用的动态 batch size、下一轮需要生成的 draft
            step，以及描述当前 proposal 布局的上一轮 draft step。
        """

        raise NotImplementedError

    @abstractmethod
    def update_statics(
        self,
        plan: SpecDecodePlan,
        proposal: SpecProposal,
        req_num: int,
        accept_lengths,
    ) -> None:
        """在本轮 verify 完成后更新规划器的运行时统计。

        Args:
            plan: 本轮 decode 实际采用的执行计划，用于确定被验证的配置。
            proposal: proposer 生成的模式专属输出。需要额外调度信息的规划器
                直接读取自己的 proposal 子类，其他规划器忽略该对象。
            req_num: 本轮逻辑请求数量。
            accept_lengths: 每个请求本轮提交的 token 数量，包含必然提交的
                target token。
        """

        raise NotImplementedError


class _InferCostMsTable:
    def __init__(self) -> None:
        self.infer_cost_ms_table = SortedDict()

    def update(self, batch_size: int, infer_cost_ms: float) -> None:
        self.infer_cost_ms_table[int(batch_size)] = float(infer_cost_ms)

    def estimate(self, batch_size: int) -> float:
        """Estimate an uncaptured batch without applying the graph-miss penalty."""

        batch_size = int(batch_size)
        max_batch_size, max_cost_ms = self.infer_cost_ms_table.peekitem(-1)
        if batch_size <= max_batch_size:
            return self._get(batch_size)
        return max_cost_ms * batch_size / max_batch_size

    def get_batch_size_keys_between(self, batch_size1: int, batch_size2: int) -> List[int]:
        start = min(int(batch_size1), int(batch_size2))
        end = max(int(batch_size1), int(batch_size2))
        batch_sizes = set(self.infer_cost_ms_table.irange(minimum=start, maximum=end, inclusive=(True, True)))
        batch_sizes.update((start, end))
        return sorted(batch_sizes)

    def _get(self, batch_size: int) -> float:
        batch_size = int(batch_size)

        if len(self.infer_cost_ms_table) == 0:
            return batch_size * 1000.0

        if batch_size in self.infer_cost_ms_table:
            return self.infer_cost_ms_table[batch_size]

        max_batch_size = self.infer_cost_ms_table.peekitem(-1)[0]
        if batch_size > max_batch_size:
            max_infer_cost_ms = self.infer_cost_ms_table.peekitem(-1)[1]
            # 超过最大 graph 范围时使用高惩罚，使调度器倾向于关闭 speculative draft。
            return max_infer_cost_ms + (batch_size - max_batch_size) * 1000.0

        index = self.infer_cost_ms_table.bisect_left(batch_size)
        return self.infer_cost_ms_table.peekitem(index)[1]


class _EMAValue:
    def __init__(self, decay: float, init_value: float) -> None:
        self.decay = decay
        self.value = init_value
        self.update_count = 0

    def update(self, new_value: float):
        self.update_count += 1
        self.value = self.decay * self.value + (1.0 - self.decay) * new_value

    def get(self) -> float:
        return self.value

    def get_count(self) -> int:
        return self.update_count

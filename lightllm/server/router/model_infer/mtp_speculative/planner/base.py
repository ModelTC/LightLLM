from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sortedcontainers import SortedDict


@dataclass(frozen=True)
class SpecDecodePlan:
    """Planner decision for one target decode iteration.

    Fixed scheduling uses the full speculative-expanded target batch:
    - dynamic_batch_size is None
    - draft_step == max_draft_step

    Dynamic speculative scheduling may compact target rows before forward:
    - dynamic_batch_size is the selected target row count
    - draft_step is the candidate length to generate after target verify
    - pre_draft_step describes the previous iteration and controls whether
      GPU verify sync can be skipped
    """

    dynamic_batch_size: Optional[int]
    draft_step: int
    pre_draft_step: int
    # False when the current batch contains requests without a proposal from
    # the previous iteration. Such an iteration does not represent one
    # well-defined LightSpec runtime configuration.
    all_reqs_have_proposals: bool = True

    @property
    def is_dynamic(self) -> bool:
        return self.dynamic_batch_size is not None

    @property
    def skip_verify_sync(self) -> bool:
        return self.is_dynamic and self.pre_draft_step == 0

    def filter_reqs(self, reqs: List, selected_row_mask_cpu) -> List:
        return [req for req, selected in zip(reqs, selected_row_mask_cpu.tolist()) if selected]


class _InferCostMsTable:
    def __init__(self) -> None:
        self.infer_cost_ms_table = SortedDict()

    def update(self, batch_size: int, infer_cost_ms: float) -> None:
        self.infer_cost_ms_table[int(batch_size)] = float(infer_cost_ms)

    def has_data(self) -> bool:
        return len(self.infer_cost_ms_table) > 0

    def get(self, batch_size: int) -> float:
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

    def estimate(self, batch_size: int) -> float:
        """Estimate an uncaptured batch without applying the graph-miss penalty."""

        batch_size = int(batch_size)
        max_batch_size, max_cost_ms = self.infer_cost_ms_table.peekitem(-1)
        if batch_size <= max_batch_size:
            return self.get(batch_size)
        return max_cost_ms * batch_size / max_batch_size

    def get_batch_size_keys_between(self, batch_size1: int, batch_size2: int) -> List[int]:
        start = min(int(batch_size1), int(batch_size2))
        end = max(int(batch_size1), int(batch_size2))
        batch_sizes = set(self.infer_cost_ms_table.irange(minimum=start, maximum=end, inclusive=(True, True)))
        batch_sizes.update((start, end))
        return sorted(batch_sizes)


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

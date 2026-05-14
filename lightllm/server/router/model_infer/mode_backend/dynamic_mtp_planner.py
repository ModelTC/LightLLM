from __future__ import annotations

import dataclasses
import math
import threading
from typing import List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from lightllm.server.router.model_infer.infer_batch import InferReq


# Development-time knobs. Keep these local while the dynamic MTP planner is being
# tuned; move the stable subset to StartArgs once the policy settles.
EMA_ALPHA = 0.2
BUDGET_SCALE = 1.0
MIN_STEP = 1
MAX_STEP = None


@dataclasses.dataclass
class MTPPlan:
    planned_steps: List[int]
    selected_mtp_indexes: List[List[int]]
    budget: int
    estimated_step: int
    b_req_mtp_start_loc: List[int]


class DynamicMTPPlanner:
    """
    Plans a uniform dynamic MTP verification length from historical acceptance.

    The plan is intentionally based on already available history so decode
    preprocessing does not have to wait for the current draft pass to finish.
    """

    def __init__(
        self,
        max_mtp_step: int,
        ema_alpha: float = EMA_ALPHA,
        budget_scale: float = BUDGET_SCALE,
        min_step: int = MIN_STEP,
        max_step: int = None,
    ) -> None:
        assert max_mtp_step >= 0
        assert 0.0 < ema_alpha <= 1.0
        assert budget_scale > 0.0
        self.max_mtp_step = max_mtp_step
        self.ema_alpha = ema_alpha
        self.budget_scale = budget_scale
        self.min_step = max(0, min(min_step, max_mtp_step))
        if max_step is None:
            max_step = max_mtp_step if MAX_STEP is None else MAX_STEP
        self.max_step = max(self.min_step, min(max_step, max_mtp_step))
        self._lock = threading.Lock()
        self._ema_max_accept_step = float(self.max_step)

    def build_plan(self, reqs: Sequence[InferReq]) -> MTPPlan:
        req_num = len(reqs)
        if req_num == 0:
            return MTPPlan(
                planned_steps=[],
                selected_mtp_indexes=[],
                budget=0,
                estimated_step=0,
                b_req_mtp_start_loc=[],
            )

        with self._lock:
            slot_limit = int(math.ceil(self._ema_max_accept_step * self.budget_scale))

        slot_limit = min(max(slot_limit, self.min_step), self.max_step)
        planned_steps = [slot_limit for _ in reqs]

        selected_mtp_indexes = [list(range(1, step + 1)) for step in planned_steps]

        start_locs = []
        cur_loc = 0
        for selected_indexes in selected_mtp_indexes:
            start_locs.append(cur_loc)
            cur_loc += 1 + len(selected_indexes)

        for req, step in zip(reqs, planned_steps):
            req.current_mtp_step = step

        return MTPPlan(
            planned_steps=planned_steps,
            selected_mtp_indexes=selected_mtp_indexes,
            budget=sum(planned_steps),
            estimated_step=slot_limit,
            b_req_mtp_start_loc=start_locs,
        )

    def update(
        self,
        reqs: Sequence[InferReq],
        mtp_accept_len_cpu,
    ) -> None:
        if not reqs:
            return

        accept_len_np = mtp_accept_len_cpu.numpy()
        max_accept_step = 0
        for req_index in range(len(reqs)):
            accept_len = int(accept_len_np[req_index])
            accept_step = max(0, accept_len - 1)
            max_accept_step = max(max_accept_step, min(accept_step, self.max_step))

        with self._lock:
            self._ema_max_accept_step = (
                self.ema_alpha * max_accept_step + (1.0 - self.ema_alpha) * self._ema_max_accept_step
            )

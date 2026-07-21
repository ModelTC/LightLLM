from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


@dataclass
class VLAActionModelOutput(ModelOutput):
    actions: torch.Tensor | None = None
    action_status: str = "IDLE"
    prefix_version: int = 0
    safe_to_release: bool = False
    policy_timing: Optional[dict[str, float]] = None
    action_timing_events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
    error_info: Optional[str] = None

    def resolve_action_timing(self) -> float | None:
        if self.action_timing_events is None:
            return None
        start_event, end_event = self.action_timing_events
        elapsed_ms = start_event.elapsed_time(end_event)
        if self.policy_timing is None:
            self.policy_timing = {}
        self.policy_timing["action_expert_ms"] = elapsed_ms
        self.action_timing_events = None
        return elapsed_ms

    def to_no_ref_tensor(self):
        if self.logits is not None:
            self.logits = tensor_to_no_ref_tensor(self.logits)
        if self.actions is not None:
            self.actions = tensor_to_no_ref_tensor(self.actions)

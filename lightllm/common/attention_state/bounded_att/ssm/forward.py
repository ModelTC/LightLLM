from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch

from lightllm.common.attention_state.commit import StateCommit
from lightllm.common.ssm_types import SSMRunConfig

if TYPE_CHECKING:
    from ..conv_ssm_state import ConvSSMState


@dataclass
class SSMForward:
    resource: "ConvSSMState"
    req_indices: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: Optional[torch.Tensor]
    accepted_token_counts: Optional[torch.Tensor]
    run_config: Optional[SSMRunConfig] = None

    def run_layer(self, logical_layer, q, k, v, a, b, a_log, bias):
        output, self.run_config = self.resource._ssm.forward(
            self.resource.get_storage_layer_index(logical_layer),
            q,
            k,
            v,
            a,
            b,
            a_log,
            bias,
            self.req_indices,
            self.cu_seqlens,
            run_config=self.run_config,
            state_indices=self.state_indices,
            accepted_token_counts=self.accepted_token_counts,
        )
        return output

    def finish(self):
        payload = self.resource._ssm.create_commit(self.run_config)
        if payload is None:
            return None
        if self.resource.state_name is None:
            raise ValueError("forward commits require a registered resource")
        return StateCommit(self.resource.state_name, payload)

    def copy_inputs(self, source: "SSMForward"):
        if self.resource is not source.resource:
            raise ValueError("cannot copy a forward for a different resource")
        copies = []
        for name in ("req_indices", "state_indices", "cu_seqlens", "accepted_token_counts"):
            target, value = getattr(self, name), getattr(source, name)
            if (target is None) != (value is None):
                raise ValueError(f"captured SSM layout changed: {name}")
            if target is not None:
                if target.shape != value.shape or target.dtype != value.dtype or target.device != value.device:
                    raise ValueError(f"captured SSM tensor layout changed: {name}")
                copies.append((target, value))
        copied = set()
        for target, value in copies:
            if target is not None and target.data_ptr() != value.data_ptr() and target.data_ptr() not in copied:
                target.copy_(value, non_blocking=True)
                copied.add(target.data_ptr())

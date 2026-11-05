from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch

from lightllm.common.attention_state.commit import StateCommit
from lightllm.common.ssm_types import SSMRunConfig

if TYPE_CHECKING:
    from .base import SSMExecutor
    from ..conv_ssm_state import ConvSSMState


@dataclass
class SSMForward:
    # 一次 forward 固定请求布局和执行配置；finish() 把配置封装成提交凭据，供接受阶段复用。
    resource: "ConvSSMState"
    executor: "SSMExecutor"
    req_indices: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: Optional[torch.Tensor]
    accepted_token_counts: Optional[torch.Tensor]
    run_config: Optional[SSMRunConfig] = None

    def run_layer(self, logical_layer, q, k, v, a, b, a_log, bias):
        output, self.run_config = self.executor.forward(
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
        payload = self.executor.create_commit(self.run_config)
        if payload is None:
            return None
        if self.resource.state_name is None:
            raise ValueError("forward commits require a registered resource")
        return StateCommit(self.resource.state_name, payload)

    def copy_inputs(self, source: "SSMForward"):
        if self.resource is not source.resource or self.executor is not source.executor:
            raise ValueError("cannot copy a forward for a different resource or executor")
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

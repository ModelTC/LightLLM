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
    """携带一次 forward 的请求布局，并生成对应的状态提交凭据。

    ``req_indices`` 是本次 forward 的 request-row 序列，可能包含 CUDA Graph
    的 ``HOLD`` padding；``state_indices`` 是 kernel 使用的物理槽位，
    ``cu_seqlens`` 描述真实 query 窗口。``run_config`` 在第一层执行时确定，
    后续层复用同一配置，``finish`` 再把它封装进 ``StateCommit``。
    """

    resource: "ConvSSMState"
    executor: "SSMExecutor"
    req_indices: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: Optional[torch.Tensor]
    accepted_token_counts: Optional[torch.Tensor]
    run_config: Optional[SSMRunConfig] = None

    def run_layer(self, logical_layer, q, k, v, a, b, a_log, bias):
        """执行一层线性注意力；logical_layer 会映射到物理权重层。"""

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
        """结束 forward，返回接受阶段需要的状态提交描述。"""

        payload = self.executor.create_commit(self.run_config)
        if payload is None:
            return None
        return StateCommit(self.resource.state_name, payload)

    def copy_inputs(self, source: "SSMForward"):
        """把动态图输入复制到已 capture 的 tensor，保留 capture 时的布局。"""

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

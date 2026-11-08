from __future__ import annotations

import torch

from lightllm.common.ssm_types import SSMRunConfig
from .base import SSMExecutor
from .export import export_indexed_state


class NativeSSMExecutor(SSMExecutor):
    """直接调用 fused recurrent kernel，并把 MTP 状态展开为物理槽位。"""

    def __init__(
        self, state: torch.Tensor, verify_width: int, *, kda: bool = False, gate_lower_bound: float = -5.0
    ) -> None:
        """创建直接写入 canonical state 的 Native executor。"""
        self.state = state
        self.slots_per_request = verify_width
        self.kda = kda
        self.gate_lower_bound = gate_lower_bound

    def decode_indices(self, reqs: torch.Tensor, speculative: bool) -> torch.Tensor:
        """返回普通 decode 或 speculative verify 对应的物理 state 索引。"""

        if not speculative:
            return reqs if self.slots_per_request == 1 else reqs * self.slots_per_request
        offsets = torch.arange(self.slots_per_request, device=reqs.device, dtype=reqs.dtype)
        return reqs[:, None] * self.slots_per_request + offsets[None, :]

    def get_accepted_state(self, req_idx: int, accepted_index: int | None = None) -> torch.Tensor:
        """返回单槽位 view，或 MTP 接受位置对应的 state 副本。"""

        if self.slots_per_request == 1:
            return self.state[:, req_idx]
        index = req_idx * self.slots_per_request + accepted_index
        return self.state.index_select(1, index.long().view(1))[:, 0]

    def export_accepted_state(
        self,
        req_indices: list[int],
        targets: list[torch.Tensor],
        accepted_indices: torch.Tensor | None = None,
        *,
        flush: bool = False,
    ) -> None:
        """把 native state 按请求和接受位置导出到调用方目标页。"""

        export_indexed_state(
            self.state,
            req_indices,
            targets,
            accepted_indices if self.slots_per_request > 1 else None,
            slots_per_request=self.slots_per_request,
            index_stride=self.state.stride(1),
        )

    def forward(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
        reqs: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        run_config: SSMRunConfig | None = None,
        *,
        state_indices: torch.Tensor | None = None,
        accepted_token_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """执行 native recurrent decode，并原地更新 fused kernel 的最终 state。"""

        state = self.state[layer]
        if self.kda:
            from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda_decode import fused_recurrent_kda

            output, _ = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                raw_gate=a.view(*q.shape[:2], -1),
                raw_beta=b.view(*q.shape[:2], -1),
                a_log=a_log,
                gate_bias=bias,
                initial_state=state,
                lower_bound=self.gate_lower_bound,
                inplace_final_state=True,
                ssm_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
                num_accepted_tokens=accepted_token_counts,
            )
        elif cu_seqlens is not None:
            from lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent import (
                mtp_fused_recurrent_gated_delta_rule,
            )

            output, _ = mtp_fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                initial_state=state,
                cu_seqlens=cu_seqlens.to(torch.long),
                ssm_state_indices=state_indices,
                ssm_state_write_indices=state_indices,
                num_accepted_tokens=accepted_token_counts,
                A_log=a_log,
                dt_bias=bias,
                a_raw=a,
                b_raw=b,
            )
        else:
            from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import fused_recurrent_gated_delta_rule

            output, _ = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                initial_state=state,
                inplace_final_state=True,
                ssm_state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
                A_log=a_log,
                dt_bias=bias,
                a_raw=a,
                b_raw=b,
            )
        return output, None

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch

from lightllm.common.attention_state.commit import StateCommit
from lightllm.common.attention_state.bounded_att.ssm.forward import SSMForward
from lightllm.common.basemodel.attention.base_att import BaseDecodeAttState, BasePrefillAttState
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
    build_dynamic_mtp_linear_att_state_params,
)


def get_resources(att_state: Any) -> tuple[Any, ...]:
    """返回当前角色下与线性 attention 算法匹配的 bounded State。"""
    return tuple(
        state
        for state in att_state.infer_state.att_state_manager.get_bounded_attention_states(
            att_state.infer_state.att_state_role
        )
        if isinstance(state, att_state.state_class)
    )


def get_resource(att_state: Any, logical_layer: int) -> Any:
    """按逻辑层号取得当前角色对应的 bounded State。"""
    return att_state.infer_state.att_state_manager.get_state(
        model_role=att_state.infer_state.att_state_role,
        state_type_filter=att_state.state_class.algorithm,
        logical_layer_id=logical_layer,
        state_class=att_state.state_class,
    )


@dataclass
class LinearPrefillAttState(BasePrefillAttState):
    state_indices: dict = field(default_factory=dict)

    def init_state(self) -> None:
        """为 prefill 建立每个 State 的请求和物理索引。"""

        self.state_indices = {
            state.state_name: state.prepare_prefill(self.infer_state.b_req_idx) for state in get_resources(self)
        }


@dataclass
class LinearDecodeAttState(BaseDecodeAttState):
    forwards: dict = field(default_factory=dict)
    _layout_checked: bool = field(default=False, init=False, repr=False)

    def get_draft_step(self) -> int:
        """返回当前模型的 MTP draft 步数。"""
        return self.backend.mtp_step

    def init_state(self) -> None:
        """根据固定或动态 MTP 布局创建每个 State 的 SSMForward。"""
        self._layout_checked = False
        draft_step = self.get_draft_step()
        infer_state = self.infer_state
        reqs, cu_seqlens = infer_state.b_req_idx, None
        if draft_step:
            if self.backend.uses_dynamic_spec_verify_layout():
                cu_seqlens, reqs = build_dynamic_mtp_linear_att_state_params(
                    b_req_idx=reqs,
                    b_mtp_index=infer_state.b_mtp_index,
                    hold_req_id=infer_state.req_manager.HOLD_REQUEST_ID,
                )
            else:
                width = draft_step + 1
                if infer_state.batch_size % width:
                    raise ValueError("fixed-layout decode requires batch_size divisible by draft_step + 1")
                cu_seqlens = torch.arange(
                    0,
                    infer_state.batch_size + 1,
                    width,
                    dtype=torch.int32,
                    device=reqs.device,
                )
                reqs = reqs.view(-1, width)[:, 0].contiguous()
        self.forwards = {state.state_name: state.prepare_decode(reqs, cu_seqlens) for state in get_resources(self)}

    def get_forward(self, logical_layer: int) -> SSMForward:
        """返回指定逻辑层共享的 forward 上下文。"""
        return self.forwards[get_resource(self, logical_layer).state_name]

    def get_state_commits(self) -> tuple[StateCommit, ...]:
        """收集各 State 在本轮 forward 产生的提交凭据。"""
        return tuple(commit for forward in self.forwards.values() if (commit := forward.finish()) is not None)

    def clone_for_warmup(self, infer_state: Any) -> LinearDecodeAttState:
        """复制 decode 状态供 CUDA Graph warmup 使用。"""
        state = super().clone_for_warmup(infer_state)
        state.forwards = {name: copy.copy(forward) for name, forward in self.forwards.items()}
        state._layout_checked = False
        return state

    def copy_for_decode_cuda_graph(self, new_state: LinearDecodeAttState) -> None:
        """把动态图输入复制到已 capture 的 CUDA Graph 状态。"""
        if not self._layout_checked:
            if self.forwards.keys() != new_state.forwards.keys():
                raise ValueError("captured linear attention resources changed")
        for name, forward in self.forwards.items():
            forward.copy_inputs(new_state.forwards[name])
        self._layout_checked = True

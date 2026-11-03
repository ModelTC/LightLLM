import copy
from dataclasses import dataclass, field

import torch

from lightllm.common.basemodel.attention.base_att import BaseDecodeAttState, BasePrefillAttState
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
    build_dynamic_mtp_linear_att_state_params,
)


def get_resources(att_state):
    return tuple(
        state
        for state in att_state.infer_state.att_state_manager.get_bounded_attention_states(
            att_state.infer_state.att_state_role
        )
        if isinstance(state, att_state.state_class)
    )


def get_resource(att_state, logical_layer):
    return att_state.infer_state.att_state_manager.get_state(
        model_role=att_state.infer_state.att_state_role,
        state_type_filter=att_state.state_class.algorithm,
        logical_layer_id=logical_layer,
        state_class=att_state.state_class,
    )


@dataclass
class LinearPrefillAttState(BasePrefillAttState):
    state_indices: dict = field(default_factory=dict)

    def init_state(self):

        self.state_indices = {
            state.state_name: state.prepare_prefill(self.infer_state.b_req_idx) for state in get_resources(self)
        }


@dataclass
class LinearDecodeAttState(BaseDecodeAttState):
    forwards: dict = field(default_factory=dict)

    def get_draft_step(self):
        return self.backend.mtp_step

    def init_state(self):
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

    def get_forward(self, logical_layer):
        return self.forwards[get_resource(self, logical_layer).state_name]

    def get_state_commits(self):
        return tuple(commit for forward in self.forwards.values() if (commit := forward.finish()) is not None)

    def clone_for_warmup(self, infer_state):
        state = super().clone_for_warmup(infer_state)
        state.forwards = {name: copy.copy(forward) for name, forward in self.forwards.items()}
        return state

    def copy_for_decode_cuda_graph(self, new_state):
        if self.forwards.keys() != new_state.forwards.keys():
            raise ValueError("captured linear attention resources changed")
        for name, forward in self.forwards.items():
            forward.copy_inputs(new_state.forwards[name])

"""KDA replay 的推理入口。

这个 backend 只服务 ReplayKDAState。维度在这里自己读取，不继承 native KDA backend。
"""

from __future__ import annotations

import dataclasses
import weakref
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention_state.bounded_att.kda.replay_kda_state import (
    ReplayKDAState,
)
from lightllm.common.basemodel.attention.base_att import (
    AttControl,
    BaseAttBackend,
    BaseDecodeAttState,
    BasePrefillAttState,
)
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp import (
    causal_conv1d_update as causal_conv1d_update_mtp,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda import (
    chunk_kda_with_fused_gate,
)
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
    build_dynamic_mtp_linear_att_state_params,
)

from .replay_kda_decode import replay

if TYPE_CHECKING:
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class ReplayKDALinearAttBackend(BaseAttBackend):
    """KDA replay 推理 backend。GLM-5-Next 在 replay 模式下只构造这一个类。"""

    def __init__(self, model):
        super().__init__(model=model)
        config = model.config["linear_attn_config"]
        self.num_heads = config["num_heads"]
        self.head_dim = config["head_dim"]
        assert self.num_heads % model.tp_world_size_ == 0
        self.tp_num_heads = self.num_heads // model.tp_world_size_
        self.tp_hidden_size = self.tp_num_heads * self.head_dim
        self.conv_kernel_size = config["short_conv_kernel_size"]
        self.lower_bound = config.get("gate_lower_bound", -5.0)
        self.mtp_step = model.args.mtp_step

    def split_qkv(self, mixed_qkv: torch.Tensor):
        return mixed_qkv.split(self.tp_hidden_size, dim=-1)

    def create_att_prefill_state(self, infer_state: "InferStateInfo"):
        return ReplayKDAPrefillAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))

    def create_att_decode_state(self, infer_state: "InferStateInfo"):
        return ReplayKDADecodeAttState(backend=self, _infer_state_weak_ref=weakref.ref(infer_state))


@dataclasses.dataclass
class ReplayKDAPrefillAttState(BasePrefillAttState):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None

    def init_state(self):
        reqs = self.infer_state.b_req_idx
        self.b_conv_buffer_idx = reqs
        self.b_ssm_buffer_idx = reqs

    def prefill_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.linear_att_prefill
        params = att_control.linear_att_prefill_dict
        layer_num = params["layer_num"]
        layer_weight = params["layer_weight"]
        backend: ReplayKDALinearAttBackend = self.backend
        state = _replay_state(self, layer_num)
        conv_states, ssm_states = state.get_layer_view(logical_layer_id=layer_num)
        conv_states = conv_states[..., : backend.conv_kernel_size - 1]
        mixed_qkv = causal_conv1d_fn(
            params["mixed_qkv"].transpose(0, 1),
            layer_weight.get_merged_kda_conv_weight(),
            bias=None,
            query_start_loc=self.infer_state.b1_cu_q_seq_len,
            cache_indices=self.b_conv_buffer_idx,
            has_initial_state=self.infer_state.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation="silu",
        ).transpose(0, 1)
        q, k, v = backend.split_qkv(mixed_qkv)
        q = q.view(1, -1, backend.tp_num_heads, backend.head_dim)
        k = k.view(1, -1, backend.tp_num_heads, backend.head_dim)
        v = v.view(1, -1, backend.tp_num_heads, backend.head_dim)
        raw_gate = params["raw_gate"].view(1, -1, backend.tp_hidden_size)
        raw_beta = params["raw_beta"].view(1, -1, backend.tp_num_heads)
        output, final_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_gate.view(1, -1, backend.tp_num_heads, backend.head_dim),
            beta=raw_beta.float().sigmoid(),
            A_log=layer_weight.linear_A_log.weight,
            g_bias=layer_weight.linear_dt_bias.weight,
            initial_state=ssm_states[self.b_ssm_buffer_idx].contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=self.infer_state.b1_cu_q_seq_len,
            safe_gate=True,
            lower_bound=backend.lower_bound,
        )
        ssm_states[self.b_ssm_buffer_idx] = final_state.to(ssm_states.dtype, copy=False)
        return output


@dataclasses.dataclass
class ReplayKDADecodeAttState(BaseDecodeAttState):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None
    b1_mtp_cu_q_seq_len: torch.Tensor = None
    b_num_accepted_tokens: torch.Tensor = None

    def init_state(self):
        role = self.infer_state.att_state_role
        matches = [
            state
            for state in self.infer_state.att_state_manager.get_bounded_attention_states(role)
            if isinstance(state, ReplayKDAState)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one ReplayKDAState for {role}")
        self._init_indices(self.backend.mtp_step, matches[0].mtp_state_index)
        self.b_ssm_buffer_idx = self.b_conv_buffer_idx

    def decode_att(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_control: AttControl = AttControl(),
        alloc_func=torch.empty,
    ):
        assert att_control.linear_att_decode
        params = att_control.linear_att_decode_dict
        layer_num = params["layer_num"]
        layer_weight = params["layer_weight"]
        backend: ReplayKDALinearAttBackend = self.backend
        state = _replay_state(self, layer_num)
        conv_states, _ = state.get_layer_view(logical_layer_id=layer_num)
        conv_kwargs = dict(bias=None, activation="silu", conv_state_indices=self.b_conv_buffer_idx)
        conv_update = causal_conv1d_update
        if backend.mtp_step > 0:
            conv_update = causal_conv1d_update_mtp
            conv_kwargs.update(
                mtp_step=backend.mtp_step,
                num_accepted_tokens=self.b_num_accepted_tokens,
                query_start_loc=self.b1_mtp_cu_q_seq_len,
            )
        mixed_qkv = conv_update(
            params["mixed_qkv"],
            conv_states,
            layer_weight.get_merged_kda_conv_weight(),
            **conv_kwargs,
        )
        q, k, v = backend.split_qkv(mixed_qkv)
        shape = (1, -1) if backend.mtp_step > 0 else (-1, 1)
        q = q.view(*shape, backend.tp_num_heads, backend.head_dim)
        k = k.view(*shape, backend.tp_num_heads, backend.head_dim)
        v = v.view(*shape, backend.tp_num_heads, backend.head_dim)
        raw_gate = params["raw_gate"].view(*shape, backend.tp_hidden_size)
        raw_beta = params["raw_beta"].view(*shape, backend.tp_num_heads)
        output, _ = replay(
            state,
            state.storage_layer(layer_num),
            q,
            k,
            v,
            raw_gate.reshape(-1, backend.tp_hidden_size),
            raw_beta.reshape(-1, backend.tp_num_heads),
            layer_weight.linear_A_log.weight,
            layer_weight.linear_dt_bias.weight,
            self.b_conv_buffer_idx,
            self.b1_mtp_cu_q_seq_len,
        )
        if backend.mtp_step == 0:
            state.finish_decode_token(layer_num, self.b_conv_buffer_idx)
        return output

    def _init_indices(self, mtp_step: int, mtp_index: torch.Tensor | None) -> None:
        infer = self.infer_state
        if mtp_step == 0:
            self.b_conv_buffer_idx = infer.b_req_idx
            self.b1_mtp_cu_q_seq_len = None
            self.b_num_accepted_tokens = None
            return
        if self.backend.uses_dynamic_spec_verify_layout():
            cu, conv_idx, accepted = build_dynamic_mtp_linear_att_state_params(
                b_req_idx=infer.b_req_idx,
                b_mtp_index=infer.b_mtp_index,
                req_to_mtp_state_index=mtp_index,
                hold_req_id=infer.req_manager.HOLD_REQUEST_ID,
            )
            self.b1_mtp_cu_q_seq_len = cu
            self.b_conv_buffer_idx = conv_idx
            self.b_num_accepted_tokens = accepted
            return
        width = mtp_step + 1
        batch_size = infer.batch_size
        if batch_size % width:
            raise ValueError(
                "KDA replay decode requires batch_size divisible by mtp_step + 1, "
                f"got batch_size={batch_size}, mtp_step={mtp_step}."
            )
        self.b1_mtp_cu_q_seq_len = torch.arange(
            0, batch_size + 1, width, dtype=torch.int32, device=infer.b_req_idx.device
        )
        self.b_conv_buffer_idx = infer.b_req_idx.view(-1, width)[:, 0].contiguous()
        self.b_num_accepted_tokens = mtp_index[self.b_conv_buffer_idx] + 1


def _replay_state(att_state, layer_num: int) -> ReplayKDAState:
    return att_state.infer_state.att_state_manager.get_state(
        model_role=att_state.infer_state.att_state_role,
        state_type_filter="kda",
        logical_layer_id=layer_num,
        state_class=ReplayKDAState,
    )

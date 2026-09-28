# SPDX-License-Identifier: Apache-2.0

"""KDA attention backend for GLM-5-Next."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

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
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda import chunk_kda_with_fused_gate
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda_decode import fused_recurrent_kda
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp import (
    causal_conv1d_update as causal_conv1d_update_mtp,
)
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
    build_dynamic_mtp_linear_att_state_params,
)

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo


class KDALinearAttBackend(BaseAttBackend):
    def __init__(self, model: "TpPartBaseModel"):
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

    def create_att_prefill_state(self, infer_state: "InferStateInfo"):
        return KDAPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo"):
        return KDADecodeAttState(backend=self, infer_state=infer_state)

    def split_qkv(self, mixed_qkv: torch.Tensor):
        return mixed_qkv.split(self.tp_hidden_size, dim=-1)


@dataclasses.dataclass
class KDAPrefillAttState(BasePrefillAttState):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None

    def init_state(self):
        self.b_conv_buffer_idx = self.infer_state.b_req_idx
        self.b_ssm_buffer_idx = self.infer_state.b_req_idx * (self.backend.mtp_step + 1)

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
        layer_weight = params["layer_weight"]
        layer_num = params["layer_num"]
        mixed_qkv = params["mixed_qkv"]
        raw_gate = params["raw_gate"]
        raw_beta = params["raw_beta"]
        backend: KDALinearAttBackend = self.backend

        conv_states, ssm_states = self.infer_state.req_manager.get_mamba_cache(layer_num)
        conv_states = conv_states[..., : backend.conv_kernel_size - 1]
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
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
        raw_gate = raw_gate.view(1, -1, backend.tp_hidden_size)
        raw_beta = raw_beta.view(1, -1, backend.tp_num_heads)

        initial_state = ssm_states[self.b_ssm_buffer_idx].contiguous()
        output, final_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_gate.view(1, -1, backend.tp_num_heads, backend.head_dim),
            beta=raw_beta.float().sigmoid(),
            A_log=layer_weight.linear_A_log.weight,
            g_bias=layer_weight.linear_dt_bias.weight,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=self.infer_state.b1_cu_q_seq_len,
            safe_gate=True,
            lower_bound=backend.lower_bound,
        )
        ssm_states[self.b_ssm_buffer_idx] = final_state.to(ssm_states.dtype, copy=False)
        return output


@dataclasses.dataclass
class KDADecodeAttState(BaseDecodeAttState):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None
    b1_mtp_cu_q_seq_len: torch.Tensor = None
    b_num_accepted_tokens: torch.Tensor = None

    def init_state(self):
        mtp_step = self.backend.mtp_step
        if mtp_step == 0:
            self._init_normal_decode_state()
        elif self.backend.uses_dynamic_spec_verify_layout():
            self._init_dynamic_mtp_decode_state(mtp_step + 1)
        else:
            self._init_fixed_mtp_decode_state(mtp_step)

    def _init_normal_decode_state(self):
        self.b_conv_buffer_idx = self.infer_state.b_req_idx
        self.b_ssm_buffer_idx = self.infer_state.b_req_idx

    def _init_dynamic_mtp_decode_state(self, mtp_size: int):
        (
            self.b1_mtp_cu_q_seq_len,
            self.b_conv_buffer_idx,
            self.b_num_accepted_tokens,
        ) = build_dynamic_mtp_linear_att_state_params(
            b_req_idx=self.infer_state.b_req_idx,
            b_mtp_index=self.infer_state.b_mtp_index,
            req_to_mtp_state_index=self.infer_state.req_manager.req_to_mtp_state_index,
            hold_req_id=self.infer_state.req_manager.HOLD_REQUEST_ID,
        )
        self._init_mtp_ssm_buffer_idx(mtp_size)

    def _init_fixed_mtp_decode_state(self, mtp_step: int):
        mtp_size = mtp_step + 1
        batch_size = self.infer_state.batch_size
        assert batch_size % mtp_size == 0, (
            "KDA fixed-layout decode requires batch_size to be divisible by mtp_step + 1, "
            f"got batch_size={batch_size}, mtp_step={mtp_step}."
        )

        att_batch_size = batch_size // mtp_size
        self.b1_mtp_cu_q_seq_len = torch.arange(
            0,
            batch_size + 1,
            mtp_size,
            dtype=torch.int32,
            device=self.infer_state.b_req_idx.device,
        )
        self.b_conv_buffer_idx = self.infer_state.b_req_idx.view(att_batch_size, mtp_size)[:, 0].contiguous()
        self.b_num_accepted_tokens = self.infer_state.req_manager.req_to_mtp_state_index[self.b_conv_buffer_idx] + 1
        self._init_mtp_ssm_buffer_idx(mtp_size)

    def _init_mtp_ssm_buffer_idx(self, mtp_size: int):
        att_batch_size = self.b_conv_buffer_idx.shape[0]
        # Each request owns mtp_size consecutive recurrent-state slots.
        b_ssm_buffer_start_idx = (self.b_conv_buffer_idx * mtp_size).view(att_batch_size, 1)
        state_offsets = torch.arange(
            mtp_size,
            device=self.infer_state.b_req_idx.device,
            dtype=self.infer_state.b_req_idx.dtype,
        ).view(1, mtp_size)
        self.b_ssm_buffer_idx = b_ssm_buffer_start_idx + state_offsets  # [att_batch_size, mtp_size]

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
        layer_weight = params["layer_weight"]
        layer_num = params["layer_num"]
        mixed_qkv = params["mixed_qkv"]
        raw_gate = params["raw_gate"]
        raw_beta = params["raw_beta"]
        backend: KDALinearAttBackend = self.backend

        conv_states, ssm_states = self.infer_state.req_manager.get_mamba_cache(layer_num)
        conv_kwargs = dict(bias=None, activation="silu", conv_state_indices=self.b_conv_buffer_idx)
        conv_update = causal_conv1d_update
        if backend.mtp_step > 0:
            conv_update = causal_conv1d_update_mtp
            conv_kwargs.update(
                mtp_step=backend.mtp_step,
                num_accepted_tokens=self.b_num_accepted_tokens,
                query_start_loc=self.b1_mtp_cu_q_seq_len,
            )
        mixed_qkv = conv_update(mixed_qkv, conv_states, layer_weight.get_merged_kda_conv_weight(), **conv_kwargs)
        q, k, v = backend.split_qkv(mixed_qkv)
        shape = (1, -1) if backend.mtp_step > 0 else (-1, 1)
        q = q.view(*shape, backend.tp_num_heads, backend.head_dim)
        k = k.view(*shape, backend.tp_num_heads, backend.head_dim)
        v = v.view(*shape, backend.tp_num_heads, backend.head_dim)
        raw_gate = raw_gate.view(*shape, backend.tp_hidden_size)
        raw_beta = raw_beta.view(*shape, backend.tp_num_heads)
        output, _ = fused_recurrent_kda(
            q=q,
            k=k,
            v=v,
            raw_gate=raw_gate,
            raw_beta=raw_beta,
            a_log=layer_weight.linear_A_log.weight,
            gate_bias=layer_weight.linear_dt_bias.weight,
            initial_state=ssm_states,
            lower_bound=backend.lower_bound,
            inplace_final_state=True,
            ssm_state_indices=self.b_ssm_buffer_idx,
            cu_seqlens=self.b1_mtp_cu_q_seq_len,
            num_accepted_tokens=self.b_num_accepted_tokens,
        )
        return output

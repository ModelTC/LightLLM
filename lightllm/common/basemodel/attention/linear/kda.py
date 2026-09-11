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
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.index import prepare_chunk_indices

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
        self.tp_projection_size = self.tp_num_heads * self.head_dim
        self.conv_kernel_size = config["short_conv_kernel_size"]
        self.lower_bound = config.get("gate_lower_bound", -5.0)

    def create_att_prefill_state(self, infer_state: "InferStateInfo"):
        return KDAPrefillAttState(backend=self, infer_state=infer_state)

    def create_att_decode_state(self, infer_state: "InferStateInfo"):
        return KDADecodeAttState(backend=self, infer_state=infer_state)

    def split_qkv(self, mixed_qkv: torch.Tensor):
        return mixed_qkv.split(self.tp_projection_size, dim=-1)

    def reshape_qkv(self, value: torch.Tensor, *, decode: bool):
        if decode:
            return value.view(-1, 1, self.tp_num_heads, self.head_dim)
        return value.view(1, -1, self.tp_num_heads, self.head_dim)


@dataclasses.dataclass
class KDAPrefillAttState(BasePrefillAttState):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None
    chunk_indices: torch.Tensor = None

    def init_state(self):
        self.b_conv_buffer_idx = self.infer_state.b_req_idx
        self.b_ssm_buffer_idx = self.infer_state.b_req_idx
        # Build variable-length chunk metadata once for all KDA layers.
        self.chunk_indices = prepare_chunk_indices(self.infer_state.b1_cu_q_seq_len, 64)

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

        q, k, v = [backend.reshape_qkv(x, decode=False) for x in backend.split_qkv(mixed_qkv)]
        raw_gate = raw_gate.view(1, -1, backend.tp_projection_size)
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
            chunk_indices=self.chunk_indices,
            safe_gate=True,
            lower_bound=backend.lower_bound,
        )
        ssm_states[self.b_ssm_buffer_idx] = final_state.to(ssm_states.dtype, copy=False)
        return output


@dataclasses.dataclass
class KDADecodeAttState(BaseDecodeAttState):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None

    def init_state(self):
        self.b_conv_buffer_idx = self.infer_state.b_req_idx
        self.b_ssm_buffer_idx = self.infer_state.b_req_idx

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
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer_weight.get_merged_kda_conv_weight(),
            bias=None,
            activation="silu",
            conv_state_indices=self.b_conv_buffer_idx,
        )
        q, k, v = [backend.reshape_qkv(x, decode=True) for x in backend.split_qkv(mixed_qkv)]
        raw_gate = raw_gate.view(-1, 1, backend.tp_projection_size)
        raw_beta = raw_beta.view(-1, 1, backend.tp_num_heads)
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
        )
        return output

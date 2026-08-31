# SPDX-License-Identifier: Apache-2.0

"""KDA attention backend for GLM-5-Next."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from ..base_att import AttControl, BaseAttBackend, BaseDecodeAttState, BasePrefillAttState
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_state_params import (
    build_dynamic_mtp_linear_att_state_params,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import (
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.index import prepare_chunk_indices
from lightllm.utils.envs_utils import get_env_start_args

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
        self.mtp_step = get_env_start_args().mtp_step

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
    seq_lens_cpu: list[int] = None

    def init_state(self):
        self.b_conv_buffer_idx = self.infer_state.b_req_idx
        self.b_ssm_buffer_idx = self.infer_state.b_req_idx * (self.backend.mtp_step + 1)
        self.seq_lens_cpu = (self.infer_state.b1_cu_q_seq_len[1:] - self.infer_state.b1_cu_q_seq_len[:-1]).tolist()
        # prepare_chunk_indices performs a GPU-to-CPU shape sync.  Build it
        # before entering CUDA Graph capture and copy its fixed-size contents
        # through BasePrefillAttState on replay.
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
        # MTP widens each request's convolution cache so the verify path can
        # retain all candidate states. Prefill only populates the canonical
        # history, whose width remains kernel_size - 1.
        if backend.mtp_step > 0:
            conv_states = conv_states[:, :, : -backend.mtp_step]
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer_weight.get_merged_kda_conv_weight(),
            bias=None,
            query_start_loc=self.infer_state.b1_cu_q_seq_len,
            cache_indices=self.b_conv_buffer_idx,
            has_initial_state=self.infer_state.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation="silu",
            seq_lens_cpu=self.seq_lens_cpu,
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
    b1_mtp_cu_q_seq_len: torch.Tensor = None
    b_num_accepted_tokens: torch.Tensor = None

    def init_state(self):
        draft_step = self.backend.model.mtp_manager.get_decode_draft_step(self.backend.model.is_mtp_draft_model)
        if draft_step == 0:
            self._init_normal_decode_state()
        elif self.backend.uses_dynamic_spec_verify_layout():
            self._init_dynamic_mtp_decode_state(draft_step + 1)
        else:
            self._init_fixed_mtp_decode_state(draft_step)

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

    def _init_fixed_mtp_decode_state(self, draft_step: int):
        mtp_size = draft_step + 1
        batch_size = self.infer_state.batch_size
        assert batch_size % mtp_size == 0, (
            "KDA fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
            f"got batch_size={batch_size}, draft_step={draft_step}."
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
        b_ssm_buffer_start_idx = (self.b_conv_buffer_idx * mtp_size).view(att_batch_size, 1)
        state_offsets = torch.arange(
            mtp_size,
            device=self.infer_state.b_req_idx.device,
            dtype=self.infer_state.b_req_idx.dtype,
        ).view(1, mtp_size)
        self.b_ssm_buffer_idx = b_ssm_buffer_start_idx + state_offsets

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
        draft_step = backend.model.mtp_manager.get_decode_draft_step(backend.model.is_mtp_draft_model)
        if draft_step > 0:
            return self._kda_mtp_kernel(
                mixed_qkv=mixed_qkv,
                raw_gate=raw_gate,
                raw_beta=raw_beta,
                conv_states=conv_states,
                ssm_states=ssm_states,
                layer_weight=layer_weight,
                draft_step=draft_step,
            )

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

    def _kda_mtp_kernel(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        layer_weight,
        draft_step: int,
    ):
        from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp import (
            causal_conv1d_update as causal_conv1d_update_mtp,
        )

        backend: KDALinearAttBackend = self.backend
        mixed_qkv = causal_conv1d_update_mtp(
            mixed_qkv,
            conv_states,
            layer_weight.get_merged_kda_conv_weight(),
            mtp_step=draft_step,
            bias=None,
            activation="silu",
            conv_state_indices=self.b_conv_buffer_idx,
            num_accepted_tokens=self.b_num_accepted_tokens,
            query_start_loc=self.b1_mtp_cu_q_seq_len,
        )

        q, k, v = [backend.reshape_qkv(x, decode=False) for x in backend.split_qkv(mixed_qkv)]
        raw_gate = raw_gate.view(1, -1, backend.tp_projection_size)
        raw_beta = raw_beta.view(1, -1, backend.tp_num_heads)
        assert self.b_ssm_buffer_idx.dim() == 2, "KDA MTP SSM buffer idx must be 2D [N, S+1]"
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
            cu_seqlens=self.b1_mtp_cu_q_seq_len.to(torch.long),
            ssm_state_indices=self.b_ssm_buffer_idx,
            ssm_state_write_indices=self.b_ssm_buffer_idx,
            num_accepted_tokens=self.b_num_accepted_tokens,
        )
        return output

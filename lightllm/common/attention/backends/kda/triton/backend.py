"""KDA attention backend for GLM-5-Next."""
from __future__ import annotations
from lightllm.common.attention.backends.base import BaseAttentionBackend
from lightllm.common.attention.api.spec import AttentionFamily
import dataclasses
from typing import TYPE_CHECKING
import torch
from lightllm.common.attention_state.bounded_att import KDAState
from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from lightllm.common.attention.backends.kda.triton.kernels.kda import chunk_kda_with_fused_gate
from lightllm.common.attention.backends.kda.triton.kernels.kda_decode import fused_recurrent_kda
from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import (
    causal_conv1d_update as causal_conv1d_update_mtp,
)
from lightllm.common.attention.backends._shared.linear.mtp_state_params import build_dynamic_mtp_linear_att_state_params

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.attention.backends.base import PreparedPlan
from lightllm.common.attention.api.backend import LayerBinding
from lightllm.common.attention.api.inputs import KDAInputs


@dataclasses.dataclass
class KDAPrefillPlan(PreparedPlan):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None


@dataclasses.dataclass
class KDADecodePlan(PreparedPlan):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None
    b1_mtp_cu_q_seq_len: torch.Tensor = None
    b_num_accepted_tokens: torch.Tensor = None


class KDALinearAttBackend(BaseAttentionBackend):
    provider = "triton"
    cache_formats = frozenset({"kda"})
    family = AttentionFamily.KDA

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

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return KDAPrefillPlan(capacity, capture, workspace)

    def _allocate_decode_plan(self, capacity, capture, workspace):
        return KDADecodePlan(capacity, capture, workspace)

    def split_qkv(self, mixed_qkv: torch.Tensor):
        return mixed_qkv.split(self.tp_hidden_size, dim=-1)

    def _update_prefill(self, plan):
        plan.b_conv_buffer_idx = plan.batch.b_req_idx
        plan.b_ssm_buffer_idx = plan.batch.b_req_idx * (self.mtp_step + 1)

    def _forward_prefill(self, layer: LayerBinding, plan, inputs: KDAInputs, *, out=None):
        mixed_qkv = inputs.mixed_qkv
        raw_gate = inputs.raw_gate
        raw_beta = inputs.raw_beta
        backend: KDALinearAttBackend = self
        conv_states, ssm_states = (layer.params.conv, layer.params.ssm)
        conv_states = conv_states[..., : backend.conv_kernel_size - 1]
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.params.conv_weight,
            bias=None,
            query_start_loc=plan.batch.b1_cu_q_seq_len,
            cache_indices=plan.b_conv_buffer_idx,
            has_initial_state=plan.batch.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation="silu",
        ).transpose(0, 1)
        q, k, v = backend.split_qkv(mixed_qkv)
        q = q.view(1, -1, backend.tp_num_heads, backend.head_dim)
        k = k.view(1, -1, backend.tp_num_heads, backend.head_dim)
        v = v.view(1, -1, backend.tp_num_heads, backend.head_dim)
        raw_gate = raw_gate.view(1, -1, backend.tp_hidden_size)
        raw_beta = raw_beta.view(1, -1, backend.tp_num_heads)
        initial_state = ssm_states[plan.b_ssm_buffer_idx].contiguous()
        output, final_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_gate.view(1, -1, backend.tp_num_heads, backend.head_dim),
            beta=raw_beta.float().sigmoid(),
            A_log=layer.params.a_log,
            g_bias=layer.params.dt_bias,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=plan.batch.b1_cu_q_seq_len,
            safe_gate=True,
            lower_bound=backend.lower_bound,
        )
        ssm_states[plan.b_ssm_buffer_idx] = final_state.to(ssm_states.dtype, copy=False)
        return output

    def _update_decode(self, plan):
        mtp_step = self.mtp_step
        if mtp_step == 0:
            self._decode_init_normal_decode_state(plan)
            return
        if self.uses_dynamic_spec_verify_layout():
            self._decode_init_dynamic_mtp_decode_state(plan, mtp_step + 1, req_to_mtp_state_index=self.mtp_state_index)
        else:
            self._decode_init_fixed_mtp_decode_state(plan, mtp_step, req_to_mtp_state_index=self.mtp_state_index)

    def _decode_init_normal_decode_state(self, plan):
        plan.b_conv_buffer_idx = plan.batch.b_req_idx
        plan.b_ssm_buffer_idx = plan.batch.b_req_idx

    def _decode_init_dynamic_mtp_decode_state(
        self, plan, mtp_size: int, *, req_to_mtp_state_index: torch.Tensor
    ) -> None:
        (
            plan.b1_mtp_cu_q_seq_len,
            plan.b_conv_buffer_idx,
            plan.b_num_accepted_tokens,
        ) = build_dynamic_mtp_linear_att_state_params(
            b_req_idx=plan.batch.b_req_idx,
            b_mtp_index=plan.batch.b_mtp_index,
            req_to_mtp_state_index=req_to_mtp_state_index,
            hold_req_id=plan.batch.hold_req_id,
        )
        self._decode_init_mtp_ssm_buffer_idx(plan, mtp_size)

    def _decode_init_fixed_mtp_decode_state(self, plan, mtp_step: int, *, req_to_mtp_state_index: torch.Tensor) -> None:
        mtp_size = mtp_step + 1
        batch_size = plan.batch.batch_size
        assert batch_size % mtp_size == 0, (
            "KDA fixed-layout decode requires batch_size to be divisible by mtp_step + 1, "
            f"got batch_size={batch_size}, mtp_step={mtp_step}."
        )
        att_batch_size = batch_size // mtp_size
        plan.b1_mtp_cu_q_seq_len = torch.arange(
            0, batch_size + 1, mtp_size, dtype=torch.int32, device=plan.batch.b_req_idx.device
        )
        plan.b_conv_buffer_idx = plan.batch.b_req_idx.view(att_batch_size, mtp_size)[:, 0].contiguous()
        plan.b_num_accepted_tokens = req_to_mtp_state_index[plan.b_conv_buffer_idx] + 1
        self._decode_init_mtp_ssm_buffer_idx(plan, mtp_size)

    def _decode_init_mtp_ssm_buffer_idx(self, plan, mtp_size: int):
        att_batch_size = plan.b_conv_buffer_idx.shape[0]
        b_ssm_buffer_start_idx = (plan.b_conv_buffer_idx * mtp_size).view(att_batch_size, 1)
        state_offsets = torch.arange(
            mtp_size, device=plan.batch.b_req_idx.device, dtype=plan.batch.b_req_idx.dtype
        ).view(1, mtp_size)
        plan.b_ssm_buffer_idx = b_ssm_buffer_start_idx + state_offsets

    def _forward_decode(self, layer: LayerBinding, plan, inputs: KDAInputs, *, out=None):
        mixed_qkv = inputs.mixed_qkv
        raw_gate = inputs.raw_gate
        raw_beta = inputs.raw_beta
        backend: KDALinearAttBackend = self
        conv_states, ssm_states = (layer.params.conv, layer.params.ssm)
        conv_kwargs = dict(bias=None, activation="silu", conv_state_indices=plan.b_conv_buffer_idx)
        conv_update = causal_conv1d_update
        if backend.mtp_step > 0:
            conv_update = causal_conv1d_update_mtp
            conv_kwargs.update(
                mtp_step=backend.mtp_step,
                num_accepted_tokens=plan.b_num_accepted_tokens,
                query_start_loc=plan.b1_mtp_cu_q_seq_len,
            )
        mixed_qkv = conv_update(mixed_qkv, conv_states, layer.params.conv_weight, **conv_kwargs)
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
            a_log=layer.params.a_log,
            gate_bias=layer.params.dt_bias,
            initial_state=ssm_states,
            lower_bound=backend.lower_bound,
            inplace_final_state=True,
            ssm_state_indices=plan.b_ssm_buffer_idx,
            cu_seqlens=plan.b1_mtp_cu_q_seq_len,
            num_accepted_tokens=plan.b_num_accepted_tokens,
        )
        return output

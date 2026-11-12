"""KDA attention backend for GLM-5-Next."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import KDAInputs
from lightllm.common.attention.api.spec import AttentionFamily
from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_update
from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import (
    causal_conv1d_update as causal_conv1d_update_mtp,
)
from lightllm.common.attention.backends._shared.linear.causal_conv1d_qkv import causal_conv1d_qkv
from lightllm.common.attention.backends._shared.linear.layout import LinearDecodeContext, update_linear_decode
from lightllm.common.attention.backends.base import AttentionExecutionContext, BaseAttentionBackend
from lightllm.common.attention.backends.kda.triton.kernels.kda import chunk_kda_with_fused_gate
from lightllm.common.attention.backends.kda.triton.kernels.kda_decode import fused_recurrent_kda

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


@dataclasses.dataclass(slots=True)
class KDAPrefillContext(AttentionExecutionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class KDADecodeContext(LinearDecodeContext):
    pass


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

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return KDAPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return KDADecodeContext(capacity, capture, workspace)

    def split_qkv(self, mixed_qkv: torch.Tensor):
        return mixed_qkv.split(self.tp_hidden_size, dim=-1)

    def _update_prefill(self, context):
        context.b_conv_buffer_idx = context.batch.b_req_idx
        context.b_ssm_buffer_idx = context.batch.b_req_idx * (self.mtp_step + 1)

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: KDAInputs, *, out=None):
        mixed_qkv = inputs.mixed_qkv
        raw_gate = inputs.raw_gate
        raw_beta = inputs.raw_beta
        backend: KDALinearAttBackend = self
        conv_states, ssm_states = (layer.params.conv, layer.params.ssm)
        conv_states = conv_states[..., : backend.conv_kernel_size - 1]
        q, k, v = causal_conv1d_qkv(
            mixed_qkv,
            layer.params.conv_weight,
            conv_states,
            context.batch.b1_cu_q_seq_len,
            context.b_conv_buffer_idx,
            context.batch.b_ready_cache_len,
            context.batch.max_q_seq_len,
        )
        raw_beta = raw_beta.view(1, -1, backend.tp_num_heads)
        initial_state = ssm_states[context.b_ssm_buffer_idx].contiguous()
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
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=context.batch.b1_cu_q_seq_len,
            safe_gate=True,
            lower_bound=backend.lower_bound,
        )
        ssm_states[context.b_ssm_buffer_idx] = final_state.to(ssm_states.dtype, copy=False)
        return output

    def _update_decode(self, context):
        update_linear_decode(context, self.mtp_step, self.uses_dynamic_spec_verify_layout(), replay=False)

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: KDAInputs, *, out=None):
        mixed_qkv = inputs.mixed_qkv
        raw_gate = inputs.raw_gate
        raw_beta = inputs.raw_beta
        backend: KDALinearAttBackend = self
        conv_states, ssm_states = (layer.params.conv, layer.params.ssm)
        conv_kwargs = dict(bias=None, activation="silu", conv_state_indices=context.b_conv_buffer_idx)
        conv_update = causal_conv1d_update
        if backend.mtp_step > 0:
            conv_update = causal_conv1d_update_mtp
            conv_kwargs.update(
                mtp_step=backend.mtp_step,
                num_accepted_tokens=context.accepted_tokens(layer.params.mtp_state_index),
                query_start_loc=context.b1_mtp_cu_q_seq_len,
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
            ssm_state_indices=context.b_ssm_buffer_idx,
            cu_seqlens=context.b1_mtp_cu_q_seq_len,
            num_accepted_tokens=(context.accepted_tokens(layer.params.mtp_state_index) if backend.mtp_step else None),
        )
        return output

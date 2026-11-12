from __future__ import annotations

import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import GDNInputs
from lightllm.common.attention.api.spec import AttentionFamily, AttentionPhase
from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_fn
from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import (
    causal_conv1d_update as causal_conv1d_update_mtp,
)
from lightllm.common.attention.backends._shared.linear.layout import LinearDecodeContext, update_linear_decode
from lightllm.common.attention.backends.base import AttentionExecutionContext, BaseAttentionBackend
from lightllm.common.attention.backends.gdn.triton.kernels.fused_gdn_gating import fused_gdn_gating
from lightllm.common.attention.backends.gdn.triton.kernels.gdn_decode_pack import conv_pack_gdn_decode_inputs
from lightllm.utils.envs_utils import get_env_start_args

from .decode import replay


@dataclasses.dataclass(slots=True)
class ReplayGDNPrefillContext(AttentionExecutionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class ReplayGDNDecodeContext(LinearDecodeContext):
    pass


class ReplayGDNTritonAttBackend(BaseAttentionBackend):
    provider = "triton"
    family = AttentionFamily.GDN
    cache_formats = frozenset({"gdn"})

    def __init__(self, model):
        super().__init__(model=model)
        from lightllm.common.attention.backends.gdn.triton.kernels.chunk import chunk_gated_delta_rule

        _configure_gdn(self, model, chunk_gated_delta_rule)

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return ReplayGDNPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return ReplayGDNDecodeContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        reqs = context.batch.b_req_idx
        context.b_conv_buffer_idx = reqs
        context.b_ssm_buffer_idx = reqs

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: GDNInputs, *, out=None):
        backend = self
        conv_states, ssm_states = layer.params.conv, layer.params.ssm
        if backend.mtp_step > 0:
            conv_states = conv_states[:, :, : -backend.mtp_step]
        mixed_qkv, z, b, a = _split_qkvzba(backend, inputs.mixed_qkvzba)
        core_attn_out = self._prefill(layer, context, mixed_qkv, conv_states, ssm_states, a, b)
        return core_attn_out, z

    def _prefill(self, layer, context, mixed_qkv, conv_states, ssm_states, a, b):
        backend = self
        g, beta = fused_gdn_gating(layer.params.a_log, a, b, layer.params.dt_bias)
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.params.conv_weight,
            bias=layer.params.conv_bias,
            query_start_loc=context.batch.b1_cu_q_seq_len,
            cache_indices=context.b_conv_buffer_idx,
            has_initial_state=context.batch.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation=backend.activation,
        ).transpose(0, 1)
        query, key, value = _rearrange_mixed_qkv(backend, mixed_qkv)
        core_attn_out, last_state = backend.prefill_kernel(
            q=query,
            k=key,
            v=value,
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=ssm_states[context.b_ssm_buffer_idx],
            output_final_state=True,
            cu_seqlens=context.batch.b1_cu_q_seq_len,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_states[context.b_ssm_buffer_idx] = last_state.to(ssm_states.dtype, copy=False)
        return core_attn_out

    def _update_decode(self, context):
        update_linear_decode(context, self.decode_draft_step, self.uses_dynamic_spec_verify_layout(), replay=True)

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: GDNInputs, *, out=None):
        backend = self
        mixed_qkv, z, b, a = _split_qkvzba(backend, inputs.mixed_qkvzba)
        state = layer.params.replay_state
        conv_states = layer.params.conv
        draft_step = backend.decode_draft_step
        if draft_step > 0:
            core = self._mtp(layer, context, mixed_qkv, conv_states, a, b, draft_step, state)
        else:
            core, z = self._decode(layer, context, mixed_qkv, z, conv_states, a, b, state)
        # 各层都读到了本步开始时的 cursor。最后一层返回后再按整窗全接受推进。
        state.finish_decode_token(layer.logical_layer_id, context.b_conv_buffer_idx, context.b_cu_seqlens)
        return core, z

    def _decode(self, layer, context, mixed_qkv, z, conv_states, a, b, state):
        backend = self
        query, key, value, z, a, b = conv_pack_gdn_decode_inputs(
            mixed_qkv,
            z,
            a,
            b,
            conv_states,
            layer.params.conv_weight,
            layer.params.conv_bias,
            context.b_conv_buffer_idx,
            backend.activation,
            backend.conv_kernel_dim,
            backend.tp_num_k_heads,
            backend.head_k_dim,
            backend.tp_num_v_heads,
            backend.head_v_dim,
        )
        query = query.view(query.shape[0], query.shape[-2], query.shape[-1])
        key = key.view(key.shape[0], key.shape[-2], key.shape[-1])
        value = value.view(value.shape[0], value.shape[-2], value.shape[-1])
        output = replay(
            state,
            state.storage_layer(layer.logical_layer_id),
            query,
            key,
            value,
            a,
            b,
            layer.params.a_log,
            layer.params.dt_bias,
            context.b_conv_buffer_idx,
            context.b_cu_seqlens,
        )
        return output, z

    def _mtp(self, layer, context, mixed_qkv, conv_states, a, b, draft_step, state):
        backend = self
        mixed_qkv = causal_conv1d_update_mtp(
            mixed_qkv,
            conv_states,
            layer.params.conv_weight,
            mtp_step=draft_step,
            bias=layer.params.conv_bias,
            activation=backend.activation,
            conv_state_indices=context.b_conv_buffer_idx,
            num_accepted_tokens=context.accepted_tokens(layer.params.mtp_state_index),
            query_start_loc=context.b1_mtp_cu_q_seq_len,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [backend.tp_key_dim, backend.tp_key_dim, backend.tp_value_dim],
            dim=-1,
        )
        query = query.view(-1, backend.tp_num_k_heads, backend.head_k_dim)
        key = key.view(-1, backend.tp_num_k_heads, backend.head_k_dim)
        value = value.view(-1, backend.tp_num_v_heads, backend.head_v_dim)
        output = replay(
            state,
            state.storage_layer(layer.logical_layer_id),
            query,
            key,
            value,
            a,
            b,
            layer.params.a_log,
            layer.params.dt_bias,
            context.b_conv_buffer_idx,
            context.b_cu_seqlens,
        )
        return output


class ReplayGDNFlashQlaAttBackend(ReplayGDNTritonAttBackend):
    provider = "flashqla"
    dependency = "flash_qla"
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    supported_phases = frozenset({AttentionPhase.PREFILL})

    def __init__(self, model):
        BaseAttentionBackend.__init__(self, model=model)
        from flash_qla import chunk_gated_delta_rule

        _configure_gdn(self, model, chunk_gated_delta_rule)


def _configure_gdn(backend, model, prefill_kernel) -> None:
    """写入 GDN 的张量并行尺寸，并绑定这个 backend 自己的 prefill kernel。"""

    config = model.config
    tp_world_size = model.tp_world_size_
    backend.mtp_step = get_env_start_args().mtp_step
    backend.num_v_heads = config["linear_num_value_heads"]
    backend.num_k_heads = config["linear_num_key_heads"]
    backend.head_k_dim = config["linear_key_head_dim"]
    backend.head_v_dim = config["linear_value_head_dim"]
    backend.key_dim = backend.head_k_dim * backend.num_k_heads
    backend.value_dim = backend.head_v_dim * backend.num_v_heads
    backend.conv_kernel_dim = config["linear_conv_kernel_dim"]
    backend.activation = config["hidden_act"]
    backend.tp_qkvz_dim = (backend.key_dim * 2 + backend.value_dim * 2) // tp_world_size
    backend.tp_ba_dim = (backend.num_v_heads * 2) // tp_world_size
    backend.tp_num_k_heads = backend.num_k_heads // tp_world_size
    backend.tp_num_v_heads = backend.num_v_heads // tp_world_size
    backend.tp_key_dim = backend.key_dim // tp_world_size
    backend.tp_value_dim = backend.value_dim // tp_world_size
    assert backend.num_v_heads % backend.num_k_heads == 0, "num_v_heads must be divisible by num_k_heads"
    backend.num_v_heads_per_k_head = backend.num_v_heads // backend.num_k_heads
    ssm_dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    backend.ssm_state_dtype = ssm_dtypes.get(get_env_start_args().linear_att_ssm_data_type, torch.bfloat16)
    backend.prefill_kernel = prefill_kernel


def _split_qkvzba(backend, mixed_qkvzba):
    qkv_dim = backend.tp_key_dim * 2 + backend.tp_value_dim
    z_end = qkv_dim + backend.tp_value_dim
    b_end = z_end + backend.tp_num_v_heads
    mixed_qkv = mixed_qkvzba[:, :qkv_dim]
    z = mixed_qkvzba[:, qkv_dim:z_end].view(-1, backend.tp_num_v_heads, backend.head_v_dim)
    b = mixed_qkvzba[:, z_end:b_end]
    a = mixed_qkvzba[:, b_end:]
    return mixed_qkv, z, b, a


def _rearrange_mixed_qkv(backend, mixed_qkv, decode=False):
    query, key, value = torch.split(
        mixed_qkv,
        [backend.tp_key_dim, backend.tp_key_dim, backend.tp_value_dim],
        dim=-1,
    )
    if decode:
        batch_size = mixed_qkv.shape[0]
        leading = (batch_size, 1)
    else:
        leading = (1, query.shape[0])
    query = query.view(*leading, backend.tp_num_k_heads, backend.head_k_dim)
    key = key.view(*leading, backend.tp_num_k_heads, backend.head_k_dim)
    value = value.view(*leading, backend.tp_num_v_heads, backend.head_v_dim)
    return query, key, value

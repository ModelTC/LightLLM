from lightllm.common.attention.api.spec import CacheEncoding, AttentionFamily, AttentionPhase, AttentionFeature
import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttention
from lightllm.common.attention.api.inputs import GDNInputs, GDNOutput
from lightllm.common.attention.backends._shared.linear.gdn import GDNMetadata
from lightllm.common.attention.backends._shared.linear.layout import LinearDecodeContext, update_linear_decode
from lightllm.common.attention.api.backend import AttentionContext, AttentionBackend


@dataclasses.dataclass(slots=True)
class GDNPrefillContext(AttentionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None


@dataclasses.dataclass(slots=True)
class GDNDecodeContext(LinearDecodeContext):
    pass


class TritonGDNCommon(GDNMetadata):
    supported_features = frozenset({AttentionFeature.MTP, AttentionFeature.DYNAMIC_VERIFY, AttentionFeature.MTP_WINDOW})

    provider = "triton"

    cache_encodings = frozenset({CacheEncoding.PLAIN})

    family = AttentionFamily.GDN

    def __init__(self, env, spec):
        super().__init__(env=env, spec=spec)
        self._init_linear_layer_metadata(network_config=env.network_config, tp_world_size=env.tp_size)
        self.state_slots = env.state_slots


class TritonGDNPrefillBackend(TritonGDNCommon, AttentionBackend[GDNPrefillContext, GDNInputs, GDNOutput]):
    phase = AttentionPhase.PREFILL

    def __init__(self, env, spec):
        super().__init__(env, spec)
        self.prefill_kernel = self.get_prefill_kernel()

    def get_prefill_kernel(self):
        from lightllm.common.attention.backends.gdn.triton.kernels.chunk import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    def _run_prefill(
        self,
        layer,
        context,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_fn
        from lightllm.common.attention.backends.gdn.triton.kernels.fused_gdn_gating import fused_gdn_gating

        g, beta = fused_gdn_gating(layer.resources.a_log, a, b, layer.resources.dt_bias)
        mixed_qkv = mixed_qkv.transpose(0, 1)
        out_tensor = causal_conv1d_fn(
            mixed_qkv,
            layer.resources.conv_weight,
            bias=layer.resources.conv_bias,
            query_start_loc=context.batch.b1_cu_q_seq_len,
            cache_indices=context.b_conv_buffer_idx,
            has_initial_state=context.batch.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation=self.activation,
        )
        mixed_qkv = out_tensor.transpose(0, 1)
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv)
        initial_state = ssm_states[context.b_ssm_buffer_idx]
        core_attn_out, last_recurrent_state = self.prefill_kernel(
            q=query,
            k=key,
            v=value,
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=context.batch.b1_cu_q_seq_len,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_states[context.b_ssm_buffer_idx] = last_recurrent_state.to(ssm_states.dtype, copy=False)
        return core_attn_out

    def allocate_context(self, capacity, *, capture, workspace) -> GDNPrefillContext:
        return GDNPrefillContext(capacity, capture, workspace)

    def build_device_metadata(self, context: GDNPrefillContext):
        mtp_step = self.state_slots - 1
        context.b_conv_buffer_idx = context.batch.b_req_idx
        context.b_ssm_buffer_idx = context.batch.b_req_idx * (mtp_step + 1)
        return

    def forward(self, layer: LayerAttention, context: GDNPrefillContext, inputs: GDNInputs, *, out=None) -> GDNOutput:
        mixed_qkvzba: torch.Tensor = inputs.mixed_qkvzba
        conv_states, ssm_states = (layer.resources.conv, layer.resources.ssm)
        if (self.state_slots - 1) > 0:
            conv_states = conv_states[:, :, : -(self.state_slots - 1)]
        mixed_qkv, z, b, a = self._split_qkvzba(mixed_qkvzba)
        core_attn_out = self._run_prefill(layer, context, mixed_qkv, conv_states, ssm_states, a, b)
        return GDNOutput(core_attn_out, z)


class TritonGDNDecodeBackend(TritonGDNCommon, AttentionBackend[GDNDecodeContext, GDNInputs, GDNOutput]):
    phase = AttentionPhase.DECODE

    def _run_decode(
        self,
        layer,
        context,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        from lightllm.common.attention.backends.gdn.triton.kernels.fused_recurrent import (
            fused_recurrent_gated_delta_rule,
        )
        from lightllm.common.attention.backends.gdn.triton.kernels.gdn_decode_pack import conv_pack_gdn_decode_inputs

        query, key, value, z, a, b = conv_pack_gdn_decode_inputs(
            mixed_qkv,
            z,
            a,
            b,
            conv_states,
            layer.resources.conv_weight,
            layer.resources.conv_bias,
            context.b_conv_buffer_idx,
            self.activation,
            self.conv_kernel_dim,
            self.tp_num_k_heads,
            self.head_k_dim,
            self.tp_num_v_heads,
            self.head_v_dim,
        )
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            initial_state=ssm_states,
            inplace_final_state=True,
            ssm_state_indices=context.b_ssm_buffer_idx,
            use_qk_l2norm_in_kernel=True,
            A_log=layer.resources.a_log,
            dt_bias=layer.resources.dt_bias,
            a_raw=a,
            b_raw=b,
        )
        return GDNOutput(core_attn_out, z)

    def _run_mtp(
        self,
        layer,
        context,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        from lightllm.common.attention.backends.gdn.triton.kernels.mtp_fused_recurrent import (
            mtp_fused_recurrent_gated_delta_rule,
        )
        from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import (
            causal_conv1d_update as causal_conv1d_update_mtp,
        )

        cu_seqlens_q = context.b1_mtp_cu_q_seq_len
        mixed_qkv = causal_conv1d_update_mtp(
            mixed_qkv,
            conv_states,
            layer.resources.conv_weight,
            mtp_step=self.draft_step,
            bias=layer.resources.conv_bias,
            activation=self.activation,
            conv_state_indices=context.b_conv_buffer_idx,
            num_accepted_tokens=context.accepted_tokens(layer.resources.mtp_state_index),
            query_start_loc=cu_seqlens_q,
        )
        query, key, value = self._rearrange_mixed_qkv(mixed_qkv, decode=False)
        assert context.b_ssm_buffer_idx.dim() == 2, "SSM buffer idx must be 2D [N, S+1]"
        core_attn_out, _ = mtp_fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            initial_state=ssm_states,
            cu_seqlens=cu_seqlens_q.to(torch.long),
            ssm_state_indices=context.b_ssm_buffer_idx,
            ssm_state_write_indices=context.b_ssm_buffer_idx,
            num_accepted_tokens=context.accepted_tokens(layer.resources.mtp_state_index),
            A_log=layer.resources.a_log,
            dt_bias=layer.resources.dt_bias,
            a_raw=a,
            b_raw=b,
        )
        return core_attn_out

    def allocate_context(self, capacity, *, capture, workspace) -> GDNDecodeContext:
        return GDNDecodeContext(capacity, capture, workspace)

    def build_device_metadata(self, context: GDNDecodeContext):
        update_linear_decode(context, self.draft_step, self.spec.dynamic_verify, replay=False)

    def forward(self, layer: LayerAttention, context: GDNDecodeContext, inputs: GDNInputs, *, out=None) -> GDNOutput:
        mixed_qkvzba = inputs.mixed_qkvzba
        mixed_qkv, z, b, a = self._split_qkvzba(mixed_qkvzba)
        conv_states, ssm_states = (layer.resources.conv, layer.resources.ssm)
        draft_step = self.draft_step
        if draft_step > 0:
            core_attn_out = self._run_mtp(layer, context, mixed_qkv, conv_states, ssm_states, a, b)
        else:
            core_attn_out, z = self._run_decode(layer, context, mixed_qkv, z, conv_states, ssm_states, a, b)
        return GDNOutput(core_attn_out, z)

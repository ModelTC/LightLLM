import dataclasses
from typing import TYPE_CHECKING

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.batch import AttentionBatch
from lightllm.common.attention.api.inputs import GDNInputs
from lightllm.common.attention.api.spec import AttentionFamily
from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_fn
from lightllm.common.attention.backends._shared.linear.mtp_state_params import build_dynamic_mtp_linear_att_state_params
from lightllm.common.attention.backends.base import AttentionExecutionContext, BaseAttentionBackend
from lightllm.common.attention.backends.gdn.triton.kernels.fused_gdn_gating import fused_gdn_gating
from lightllm.common.attention.backends.gdn.triton.kernels.fused_recurrent import fused_recurrent_gated_delta_rule
from lightllm.common.attention.backends.gdn.triton.kernels.gdn_decode_pack import conv_pack_gdn_decode_inputs
from lightllm.common.attention.backends.gdn.triton.kernels.mtp_fused_recurrent import (
    mtp_fused_recurrent_gated_delta_rule,
)
from lightllm.utils.envs_utils import get_env_start_args

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel


@dataclasses.dataclass
class LinearAttPrefillContext(AttentionExecutionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None


@dataclasses.dataclass
class LinearAttDecodeContext(AttentionExecutionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None
    b1_mtp_cu_q_seq_len: torch.Tensor = None
    b_num_accepted_tokens: torch.Tensor = None


class TritonLinearAttBackend(BaseAttentionBackend):
    provider = "triton"
    cache_formats = frozenset({"gdn"})
    family = AttentionFamily.GDN

    def __init__(self, model: "TpPartBaseModel"):
        super().__init__(model=model)
        self._init_linear_layer_metadata(network_config=model.config, tp_world_size=model.tp_world_size_)
        self.prefill_kernel = self.get_prefill_kernel()

    def get_prefill_kernel(self):
        from lightllm.common.attention.backends.gdn.triton.kernels.chunk import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    def _init_linear_layer_metadata(self, network_config, tp_world_size):
        self.mtp_step = get_env_start_args().mtp_step
        self.num_v_heads = network_config["linear_num_value_heads"]
        self.num_k_heads = network_config["linear_num_key_heads"]
        self.head_k_dim = network_config["linear_key_head_dim"]
        self.head_v_dim = network_config["linear_value_head_dim"]
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_dim = network_config["linear_conv_kernel_dim"]
        self.activation = network_config["hidden_act"]
        self.tp_qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) // tp_world_size
        self.tp_ba_dim = self.num_v_heads * 2 // tp_world_size
        self.tp_num_k_heads = self.num_k_heads // tp_world_size
        self.tp_num_v_heads = self.num_v_heads // tp_world_size
        self.tp_key_dim = self.key_dim // tp_world_size
        self.tp_value_dim = self.value_dim // tp_world_size
        assert self.num_v_heads % self.num_k_heads == 0, "num_v_heads must be divisible by num_k_heads"
        self.num_v_heads_per_k_head = self.num_v_heads // self.num_k_heads
        ssm_dtype_dict = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        self.ssm_state_dtype = ssm_dtype_dict.get(get_env_start_args().linear_att_ssm_data_type, torch.bfloat16)
        return

    def _split_qkvzba(self, mixed_qkvzba):
        qkv_dim = self.tp_key_dim * 2 + self.tp_value_dim
        z_end = qkv_dim + self.tp_value_dim
        b_end = z_end + self.tp_num_v_heads
        mixed_qkv = mixed_qkvzba[:, :qkv_dim]
        z = mixed_qkvzba[:, qkv_dim:z_end].view(-1, self.tp_num_v_heads, self.head_v_dim)
        b = mixed_qkvzba[:, z_end:b_end]
        a = mixed_qkvzba[:, b_end:]
        return (mixed_qkv, z, b, a)

    def _rearrange_mixed_qkv(self, mixed_qkv, decode=False):
        if decode:
            query, key, value = torch.split(mixed_qkv, [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim], dim=-1)
            batch_size = mixed_qkv.shape[0]
            query = query.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(batch_size, 1, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(batch_size, 1, self.tp_num_v_heads, self.head_v_dim)
            return (query, key, value)
        else:
            query, key, value = torch.split(mixed_qkv, [self.tp_key_dim, self.tp_key_dim, self.tp_value_dim], dim=-1)
            seq_len = query.shape[0]
            query = query.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
            key = key.view(1, seq_len, self.tp_num_k_heads, self.head_k_dim)
            value = value.view(1, seq_len, self.tp_num_v_heads, self.head_v_dim)
            return (query, key, value)

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return LinearAttPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return LinearAttDecodeContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        backend: TritonLinearAttBackend = self
        mtp_step = backend.mtp_step
        context.b_conv_buffer_idx = context.batch.b_req_idx
        context.b_ssm_buffer_idx = context.batch.b_req_idx * (mtp_step + 1)
        return

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: GDNInputs, *, out=None) -> torch.Tensor:
        mixed_qkvzba: torch.Tensor = inputs.mixed_qkvzba
        backend: TritonLinearAttBackend = self
        conv_states, ssm_states = (layer.params.conv, layer.params.ssm)
        if backend.mtp_step > 0:
            conv_states = conv_states[:, :, : -backend.mtp_step]
        mixed_qkv, z, b, a = backend._split_qkvzba(mixed_qkvzba)
        core_attn_out = self._prefill_gdn_prefill_kernel(
            layer, context, mixed_qkv, conv_states, ssm_states, a, b, context.batch
        )
        return (core_attn_out, z)

    def _prefill_gdn_prefill_kernel(
        self,
        layer,
        context,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        batch: AttentionBatch,
    ):
        g, beta = fused_gdn_gating(layer.params.a_log, a, b, layer.params.dt_bias)
        mixed_qkv = mixed_qkv.transpose(0, 1)
        backend: TritonLinearAttBackend = self
        out_tensor = causal_conv1d_fn(
            mixed_qkv,
            layer.params.conv_weight,
            bias=layer.params.conv_bias,
            query_start_loc=batch.b1_cu_q_seq_len,
            cache_indices=context.b_conv_buffer_idx,
            has_initial_state=batch.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation=backend.activation,
        )
        mixed_qkv = out_tensor.transpose(0, 1)
        query, key, value = backend._rearrange_mixed_qkv(mixed_qkv)
        initial_state = ssm_states[context.b_ssm_buffer_idx]
        core_attn_out, last_recurrent_state = backend.prefill_kernel(
            q=query,
            k=key,
            v=value,
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=batch.b1_cu_q_seq_len,
            use_qk_l2norm_in_kernel=True,
        )
        ssm_states[context.b_ssm_buffer_idx] = last_recurrent_state.to(ssm_states.dtype, copy=False)
        return core_attn_out

    def _update_decode(self, context):
        draft_step = self.decode_draft_step
        if draft_step == 0:
            self._decode_init_normal_decode_state(context)
            return
        if self.uses_dynamic_spec_verify_layout():
            self._decode_init_dynamic_mtp_decode_state(
                context, draft_step + 1, req_to_mtp_state_index=self.mtp_state_index
            )
        else:
            self._decode_init_fixed_mtp_decode_state(context, draft_step, req_to_mtp_state_index=self.mtp_state_index)

    def _decode_init_normal_decode_state(self, context):
        context.b_conv_buffer_idx = context.batch.b_req_idx
        context.b_ssm_buffer_idx = context.batch.b_req_idx

    def _decode_init_dynamic_mtp_decode_state(
        self, context, mtp_size: int, *, req_to_mtp_state_index: torch.Tensor
    ) -> None:
        (
            context.b1_mtp_cu_q_seq_len,
            context.b_conv_buffer_idx,
            context.b_num_accepted_tokens,
        ) = build_dynamic_mtp_linear_att_state_params(
            b_req_idx=context.batch.b_req_idx,
            b_mtp_index=context.batch.b_mtp_index,
            req_to_mtp_state_index=req_to_mtp_state_index,
            hold_req_id=context.batch.hold_req_id,
        )
        self._decode_init_mtp_ssm_buffer_idx(context, mtp_size)

    def _decode_init_fixed_mtp_decode_state(
        self, context, draft_step: int, *, req_to_mtp_state_index: torch.Tensor
    ) -> None:
        mtp_size = draft_step + 1
        batch_size = context.batch.batch_size
        assert batch_size % mtp_size == 0, (
            "GDN fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
            f"got batch_size={batch_size}, draft_step={draft_step}."
        )
        att_batch_size = batch_size // mtp_size
        context.b1_mtp_cu_q_seq_len = torch.arange(
            0, batch_size + 1, mtp_size, dtype=torch.int32, device=context.batch.b_req_idx.device
        )
        context.b_conv_buffer_idx = context.batch.b_req_idx.view(att_batch_size, mtp_size)[:, 0].contiguous()
        context.b_num_accepted_tokens = req_to_mtp_state_index[context.b_conv_buffer_idx] + 1
        self._decode_init_mtp_ssm_buffer_idx(context, mtp_size)

    def _decode_init_mtp_ssm_buffer_idx(self, context, mtp_size: int):
        att_batch_size = context.b_conv_buffer_idx.shape[0]
        b_ssm_buffer_start_idx = (context.b_conv_buffer_idx * mtp_size).view(att_batch_size, 1)
        state_offsets = torch.arange(
            mtp_size, device=context.batch.b_req_idx.device, dtype=context.batch.b_req_idx.dtype
        ).view(1, mtp_size)
        context.b_ssm_buffer_idx = b_ssm_buffer_start_idx + state_offsets

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: GDNInputs, *, out=None) -> torch.Tensor:
        mixed_qkvzba = inputs.mixed_qkvzba
        backend: TritonLinearAttBackend = self
        mixed_qkv, z, b, a = backend._split_qkvzba(mixed_qkvzba)
        conv_states, ssm_states = (layer.params.conv, layer.params.ssm)
        draft_step = self.decode_draft_step
        if draft_step > 0:
            core_attn_out = self._decode_gdn_mtp_kernel(
                layer, context, mixed_qkv, conv_states, ssm_states, a, b, context.batch
            )
        else:
            core_attn_out, z = self._decode_gdn_decode_kernel(
                layer, context, mixed_qkv, z, conv_states, ssm_states, a, b, context.batch
            )
        return (core_attn_out, z)

    def _decode_gdn_decode_kernel(
        self,
        layer,
        context,
        mixed_qkv: torch.Tensor,
        z: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        batch: AttentionBatch,
    ):
        backend: TritonLinearAttBackend = self
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
        core_attn_out, _ = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            initial_state=ssm_states,
            inplace_final_state=True,
            ssm_state_indices=context.b_ssm_buffer_idx,
            use_qk_l2norm_in_kernel=True,
            A_log=layer.params.a_log,
            dt_bias=layer.params.dt_bias,
            a_raw=a,
            b_raw=b,
        )
        return (core_attn_out, z)

    def _decode_gdn_mtp_kernel(
        self,
        layer,
        context,
        mixed_qkv: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        batch: AttentionBatch,
    ):
        from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import (
            causal_conv1d_update as causal_conv1d_update_mtp,
        )

        backend: TritonLinearAttBackend = self
        cu_seqlens_q = context.b1_mtp_cu_q_seq_len
        mixed_qkv = causal_conv1d_update_mtp(
            mixed_qkv,
            conv_states,
            layer.params.conv_weight,
            mtp_step=backend.decode_draft_step,
            bias=layer.params.conv_bias,
            activation=backend.activation,
            conv_state_indices=context.b_conv_buffer_idx,
            num_accepted_tokens=context.b_num_accepted_tokens,
            query_start_loc=cu_seqlens_q,
        )
        query, key, value = backend._rearrange_mixed_qkv(mixed_qkv, decode=False)
        assert context.b_ssm_buffer_idx.dim() == 2, "SSM buffer idx must be 2D [N, S+1]"
        core_attn_out, _ = mtp_fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            initial_state=ssm_states,
            cu_seqlens=cu_seqlens_q.to(torch.long),
            ssm_state_indices=context.b_ssm_buffer_idx,
            ssm_state_write_indices=context.b_ssm_buffer_idx,
            num_accepted_tokens=context.b_num_accepted_tokens,
            A_log=layer.params.a_log,
            dt_bias=layer.params.dt_bias,
            a_raw=a,
            b_raw=b,
        )
        return core_attn_out

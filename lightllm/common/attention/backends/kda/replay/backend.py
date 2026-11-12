from __future__ import annotations

import dataclasses

import torch

from lightllm.common.attention.api.backend import LayerAttentionBinding
from lightllm.common.attention.api.inputs import KDAInputs
from lightllm.common.attention.api.spec import AttentionFamily
from lightllm.common.attention.backends._shared.linear.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp import (
    causal_conv1d_update as causal_conv1d_update_mtp,
)
from lightllm.common.attention.backends._shared.linear.mtp_state_params import build_dynamic_mtp_linear_att_state_params
from lightllm.common.attention.backends.base import AttentionExecutionContext, BaseAttentionBackend
from lightllm.common.attention.backends.kda.triton.kernels.kda import chunk_kda_with_fused_gate

from .decode import replay


@dataclasses.dataclass
class ReplayKDAPrefillContext(AttentionExecutionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None


@dataclasses.dataclass
class ReplayKDADecodeContext(AttentionExecutionContext):
    b_conv_buffer_idx: torch.Tensor = None
    b_ssm_buffer_idx: torch.Tensor = None
    b1_mtp_cu_q_seq_len: torch.Tensor = None
    b_cu_seqlens: torch.Tensor = None
    b_num_accepted_tokens: torch.Tensor = None


class ReplayKDALinearAttBackend(BaseAttentionBackend):
    provider = "triton"
    family = AttentionFamily.KDA
    cache_formats = frozenset({"kda"})

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

    def _allocate_prefill_context(self, capacity, capture, workspace):
        return ReplayKDAPrefillContext(capacity, capture, workspace)

    def _allocate_decode_context(self, capacity, capture, workspace):
        return ReplayKDADecodeContext(capacity, capture, workspace)

    def _update_prefill(self, context):
        reqs = context.batch.b_req_idx
        context.b_conv_buffer_idx = reqs
        context.b_ssm_buffer_idx = reqs

    def _forward_prefill(self, layer: LayerAttentionBinding, context, inputs: KDAInputs, *, out=None):
        backend: ReplayKDALinearAttBackend = self
        state = layer.params.state
        conv_states, ssm_states = state.get_layer_view(logical_layer_id=layer.logical_layer_id)
        conv_states = conv_states[..., : backend.conv_kernel_size - 1]
        mixed_qkv = causal_conv1d_fn(
            inputs.mixed_qkv.transpose(0, 1),
            layer.params.conv_weight,
            bias=None,
            query_start_loc=context.batch.b1_cu_q_seq_len,
            cache_indices=context.b_conv_buffer_idx,
            has_initial_state=context.batch.b_ready_cache_len > 0,
            conv_states=conv_states,
            activation="silu",
        ).transpose(0, 1)
        q, k, v = backend.split_qkv(mixed_qkv)
        q = q.view(1, -1, backend.tp_num_heads, backend.head_dim)
        k = k.view(1, -1, backend.tp_num_heads, backend.head_dim)
        v = v.view(1, -1, backend.tp_num_heads, backend.head_dim)
        raw_gate = inputs.raw_gate.view(1, -1, backend.tp_hidden_size)
        raw_beta = inputs.raw_beta.view(1, -1, backend.tp_num_heads)
        output, final_state = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=raw_gate.view(1, -1, backend.tp_num_heads, backend.head_dim),
            beta=raw_beta.float().sigmoid(),
            A_log=layer.params.a_log,
            g_bias=layer.params.dt_bias,
            initial_state=ssm_states[context.b_ssm_buffer_idx].contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=context.batch.b1_cu_q_seq_len,
            safe_gate=True,
            lower_bound=backend.lower_bound,
        )
        ssm_states[context.b_ssm_buffer_idx] = final_state.to(ssm_states.dtype, copy=False)
        return output

    def _update_decode(self, context):
        self._init_indices(context, self.decode_draft_step, self.mtp_state_index)
        context.b_ssm_buffer_idx = context.b_conv_buffer_idx

    def _forward_decode(self, layer: LayerAttentionBinding, context, inputs: KDAInputs, *, out=None):
        backend: ReplayKDALinearAttBackend = self
        state = layer.params.state
        conv_states, _ = state.get_layer_view(logical_layer_id=layer.logical_layer_id)
        conv_kwargs = dict(bias=None, activation="silu", conv_state_indices=context.b_conv_buffer_idx)
        conv_update = causal_conv1d_update
        if backend.decode_draft_step > 0:
            conv_update = causal_conv1d_update_mtp
            conv_kwargs.update(
                mtp_step=backend.decode_draft_step,
                num_accepted_tokens=context.b_num_accepted_tokens,
                query_start_loc=context.b1_mtp_cu_q_seq_len,
            )
        mixed_qkv = conv_update(
            inputs.mixed_qkv,
            conv_states,
            layer.params.conv_weight,
            **conv_kwargs,
        )
        q, k, v = backend.split_qkv(mixed_qkv)
        q = q.view(-1, backend.tp_num_heads, backend.head_dim)
        k = k.view(-1, backend.tp_num_heads, backend.head_dim)
        v = v.view(-1, backend.tp_num_heads, backend.head_dim)
        raw_gate = inputs.raw_gate.reshape(-1, backend.tp_hidden_size)
        raw_beta = inputs.raw_beta.reshape(-1, backend.tp_num_heads)
        storage_layer = state.storage_layer(layer.logical_layer_id)
        output = replay(
            state_ssm=state.ssm_state[storage_layer],
            state_keys=state.keys[storage_layer],
            state_deltas=state.deltas[storage_layer],
            state_gates=state.gates[storage_layer],
            state_raw_keys=state.raw_keys[storage_layer],
            state_raw_values=state.raw_values[storage_layer],
            state_betas=state.betas[storage_layer],
            state_cursors=state.cursors,
            state_tile_sync_mark=state.tile_sync_mark,
            q=q,
            k=k,
            v=v,
            a=raw_gate,
            b=raw_beta,
            a_log=layer.params.a_log,
            bias=layer.params.dt_bias,
            reqs=context.b_conv_buffer_idx,
            cu_seqlens=context.b_cu_seqlens,
            state_verify_width=state.verify_width,
            state_compute_mode=state.compute_mode,
            state_gate_lower_bound=state.gate_lower_bound,
        )
        # 各层都读到了本步开始时的 cursor。最后一层返回后再按整窗全接受推进。
        state.finish_decode_token(layer.logical_layer_id, context.b_conv_buffer_idx, context.b_cu_seqlens)
        return output

    def _init_indices(self, context, mtp_step: int, mtp_index: torch.Tensor | None) -> None:
        infer = context.batch
        if mtp_step == 0:
            context.b_conv_buffer_idx = infer.b_req_idx
            context.b1_mtp_cu_q_seq_len = None
            context.b_num_accepted_tokens = None
            # 每行一个 token。replay 和 cursor 共用这份前缀和。
            context.b_cu_seqlens = torch.arange(
                infer.b_req_idx.numel() + 1, dtype=torch.int32, device=infer.b_req_idx.device
            )
            return
        if self.uses_dynamic_spec_verify_layout():
            cu, conv_idx, accepted = build_dynamic_mtp_linear_att_state_params(
                b_req_idx=infer.b_req_idx,
                b_mtp_index=infer.b_mtp_index,
                req_to_mtp_state_index=mtp_index,
                hold_req_id=infer.hold_req_id,
            )
            context.b1_mtp_cu_q_seq_len = cu
            context.b_cu_seqlens = cu
            context.b_conv_buffer_idx = conv_idx
            context.b_num_accepted_tokens = accepted
            return
        width = mtp_step + 1
        batch_size = infer.batch_size
        if batch_size % width:
            raise ValueError(
                "KDA replay decode requires batch_size divisible by mtp_step + 1, "
                f"got batch_size={batch_size}, mtp_step={mtp_step}."
            )
        context.b1_mtp_cu_q_seq_len = torch.arange(
            0, batch_size + 1, width, dtype=torch.int32, device=infer.b_req_idx.device
        )
        context.b_cu_seqlens = context.b1_mtp_cu_q_seq_len
        context.b_conv_buffer_idx = infer.b_req_idx.view(-1, width)[:, 0].contiguous()
        context.b_num_accepted_tokens = mtp_index[context.b_conv_buffer_idx] + 1

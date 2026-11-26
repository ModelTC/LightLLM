"""GDN implementations: chunked Triton prefill (flashqla.py subclasses it) and decode over native or replay
recurrent state.

Replay decode appends rows to a per-layer history that the replay kernel reads from each request's
cursor, instead of updating SSM states every token. Prefill is the regular chunked prefill with one
SSM slot per request.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.backend import AttentionBackend, TensorAllocator
from lightllm.common.basemodel.infer_struct import InferStateInfo
from . import GDNLayerConfig, GDNLayerInputs, _split_qkv


@dataclass(frozen=True)
class GDNPrefillAttState:
    """Packed prefill: request i's tokens start at cu_seqlens[i]; its conv state is slot conv_idx[i] and
    its SSM state slot ssm_idx[i]; has_initial_state[i] says whether it continues earlier tokens."""

    conv_idx: torch.Tensor
    ssm_idx: torch.Tensor
    cu_seqlens: torch.Tensor
    has_initial_state: torch.Tensor


@dataclass(frozen=True)
class GDNDecodeAttState:
    """Decode rows grouped into requests.

    Without speculative decoding (draft_step 0) each row is a request with conv and SSM slots
    conv_idx/ssm_idx, and cu_seqlens is None for native state. With draft_step > 0, cu_seqlens gives
    each request's rows, conv_idx its conv slot, num_accepted_tokens how many of the previous step's
    drafts it accepted, and ssm_idx its SSM slots ([requests, draft_step + 1] for native state, which
    keeps one per draft; [requests] for replay state). Replay decode always has cu_seqlens.
    """

    conv_idx: torch.Tensor
    ssm_idx: torch.Tensor
    cu_seqlens: Optional[torch.Tensor]
    num_accepted_tokens: Optional[torch.Tensor]
    draft_step: int


def build_recurrent_decode_state(backend: AttentionBackend, infer_state: InferStateInfo) -> GDNDecodeAttState:
    """Group decode rows into requests for a GDN or KDA decode backend (its layer config has replay).

    A CUDA graph step keeps derived tensors per (microbatch, batch size), rewritten in place.
    """
    from lightllm.common.attention.kernels.gdn.mtp_state_params import build_dynamic_mtp_linear_att_state_params

    replay = backend.layer_config.replay
    req, rows = infer_state.b_req_idx, infer_state.decode_rows_per_request
    if rows == 1:
        # Each row is its own request; replay decode still reads per-request row offsets.
        cu = None
        if replay:
            cu = torch.arange(req.numel() + 1, dtype=torch.int32, device=req.device)
            cu = backend._keep_for_graph(infer_state, "cu", cu)
        return GDNDecodeAttState(conv_idx=req, ssm_idx=req, cu_seqlens=cu, num_accepted_tokens=None, draft_step=0)

    assert infer_state.req_to_mtp_state_index is not None, "speculative GDN decode needs req_to_mtp_state_index"
    if infer_state.dynamic_decode_rows:
        cu, conv_idx, accepted = build_dynamic_mtp_linear_att_state_params(
            b_req_idx=req,
            b_mtp_index=infer_state.b_mtp_index,
            req_to_mtp_state_index=infer_state.req_to_mtp_state_index,
            hold_req_id=infer_state.pad_req_idx,
        )
    else:
        assert infer_state.batch_size % rows == 0, (
            "GDN fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
            f"got batch_size={infer_state.batch_size}, draft_step={rows - 1}."
        )
        cu = torch.arange(0, infer_state.batch_size + 1, rows, dtype=torch.int32, device=req.device)
        conv_idx = req.view(-1, rows)[:, 0].contiguous()
        accepted = infer_state.req_to_mtp_state_index[conv_idx] + 1
    cu = backend._keep_for_graph(infer_state, "cu", cu)
    conv_idx = backend._keep_for_graph(infer_state, "conv_idx", conv_idx)
    accepted = backend._keep_for_graph(infer_state, "accepted", accepted)

    if replay:
        ssm_idx = conv_idx
    else:
        # Each request owns rows consecutive recurrent-state slots.
        offsets = torch.arange(rows, device=req.device, dtype=req.dtype).view(1, rows)
        ssm_idx = backend._keep_for_graph(infer_state, "ssm_idx", (conv_idx * rows).view(-1, 1) + offsets)
    return GDNDecodeAttState(
        conv_idx=conv_idx, ssm_idx=ssm_idx, cu_seqlens=cu, num_accepted_tokens=accepted, draft_step=rows - 1
    )


def _pack_decode_inputs(config: GDNLayerConfig, inputs: GDNLayerInputs, conv_idx: torch.Tensor):
    from lightllm.common.attention.kernels.gdn.gdn_decode_pack import conv_pack_gdn_decode_inputs

    # Fused conv update and q/k/v/gate packing, without materializing the post-conv qkv.
    return conv_pack_gdn_decode_inputs(
        inputs.mixed_qkv,
        inputs.a,
        inputs.b,
        inputs.conv_states,
        inputs.conv_weight,
        inputs.conv_bias,
        conv_idx,
        config.activation,
        config.conv_kernel_dim,
        config.num_k_heads,
        config.head_k_dim,
        config.num_v_heads,
        config.head_v_dim,
    )


def _mtp_conv(config: GDNLayerConfig, inputs: GDNLayerInputs, att_state: GDNDecodeAttState):
    from lightllm.common.attention.kernels.gdn.causal_conv1d_mtp import causal_conv1d_update

    return causal_conv1d_update(
        inputs.mixed_qkv,
        inputs.conv_states,
        inputs.conv_weight,
        mtp_step=att_state.draft_step,
        bias=inputs.conv_bias,
        activation=config.activation,
        conv_state_indices=att_state.conv_idx,
        num_accepted_tokens=att_state.num_accepted_tokens,
        query_start_loc=att_state.cu_seqlens,
    )


class TritonGDNPrefillBackend(AttentionBackend[GDNLayerConfig, GDNPrefillAttState, GDNLayerInputs]):
    name = "triton_gdn_prefill"

    def __init__(self, layer_config: GDNLayerConfig) -> None:
        super().__init__(layer_config)
        # Resolve the chunk kernel at model load so auto-selection can skip unavailable backends.
        try:
            self.chunk_gated_delta_rule = self.load_chunk_kernel()
        except Exception as e:
            raise NotImplementedError(f"{type(self).__name__} cannot load its chunk kernel: {e!r}") from e

    @staticmethod
    def load_chunk_kernel():
        from lightllm.common.attention.kernels.gdn.third_party_fla.ops.chunk import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    @classmethod
    def group_key(cls, layer_config: GDNLayerConfig):
        return layer_config.replay, layer_config.state_mtp_step

    def build(self, infer_state: InferStateInfo) -> GDNPrefillAttState:
        req, config = infer_state.b_req_idx, self.layer_config
        # Native state keeps a slot per speculative row; replay state keeps one per request.
        slots_per_request = 1 if config.replay else config.state_mtp_step + 1
        return GDNPrefillAttState(
            conv_idx=req,
            ssm_idx=req if slots_per_request == 1 else req * slots_per_request,
            cu_seqlens=infer_state.b1_cu_q_seq_len,
            has_initial_state=infer_state.b_ready_cache_len > 0,
        )

    def forward(
        self,
        att_state: GDNPrefillAttState,
        layer_inputs: GDNLayerInputs,
        *,
        out: Optional[torch.Tensor],
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Return the chunk output, copying into out if supplied, and update the conv and SSM states in place."""
        from lightllm.common.attention.kernels.gdn.causal_conv1d import causal_conv1d_fn
        from lightllm.common.attention.kernels.gdn.fused_gdn_gating import fused_gdn_gating

        config, inputs = self.layer_config, layer_inputs
        g, beta = fused_gdn_gating(inputs.A_log, inputs.a, inputs.b, inputs.dt_bias)
        conv_states = inputs.conv_states
        if config.state_mtp_step > 0:
            conv_states = conv_states[:, :, : -config.state_mtp_step]
        mixed_qkv = causal_conv1d_fn(
            inputs.mixed_qkv.transpose(0, 1),
            inputs.conv_weight,
            bias=inputs.conv_bias,
            query_start_loc=att_state.cu_seqlens,
            cache_indices=att_state.conv_idx,
            has_initial_state=att_state.has_initial_state,
            conv_states=conv_states,
            activation=config.activation,
        ).transpose(0, 1)
        query, key, value = _split_qkv(config, mixed_qkv)
        tokens = query.shape[0]
        ssm_states = inputs.ssm_states
        core, last_state = self.chunk_gated_delta_rule(
            q=query.view(1, tokens, config.num_k_heads, config.head_k_dim),
            k=key.view(1, tokens, config.num_k_heads, config.head_k_dim),
            v=value.view(1, tokens, config.num_v_heads, config.head_v_dim),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=ssm_states[att_state.ssm_idx],
            output_final_state=True,
            cu_seqlens=att_state.cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        # The chunk kernel accumulates in float32; index_put does not convert dtypes implicitly.
        ssm_states[att_state.ssm_idx] = last_state.to(ssm_states.dtype, copy=False)
        output = core.view(tokens, config.num_v_heads, config.head_v_dim)
        if out is None:
            return output
        out.copy_(output)
        return out


class TritonGDNDecodeBackend(AttentionBackend[GDNLayerConfig, GDNDecodeAttState, GDNLayerInputs]):
    name = "triton_gdn_decode"
    # Which state layout the backend runs over; see GDNLayerConfig.replay.
    replay = False

    @classmethod
    def _check_supported(cls, layer_config: GDNLayerConfig) -> None:
        if layer_config.replay != cls.replay:
            state = "replay" if layer_config.replay else "native"
            raise NotImplementedError(f"{cls.name} does not run over {state} state")

    @classmethod
    def group_key(cls, layer_config: GDNLayerConfig):
        return layer_config.replay

    def build(self, infer_state: InferStateInfo) -> GDNDecodeAttState:
        return build_recurrent_decode_state(self, infer_state)

    def forward(
        self,
        att_state: GDNDecodeAttState,
        layer_inputs: GDNLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [rows, num_v_heads, head_v_dim]; update the conv and SSM states in place."""
        config, inputs = self.layer_config, layer_inputs
        if att_state.draft_step == 0:
            from lightllm.common.attention.kernels.gdn.third_party_fla.ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule,
            )

            query, key, value, a, b = _pack_decode_inputs(config, inputs, att_state.conv_idx)
            fused_recurrent_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                initial_state=inputs.ssm_states,
                inplace_final_state=True,
                ssm_state_indices=att_state.ssm_idx,
                use_qk_l2norm_in_kernel=True,
                A_log=inputs.A_log,
                dt_bias=inputs.dt_bias,
                a_raw=a,
                b_raw=b,
                out=out.view(value.shape),
            )
        else:
            from lightllm.common.attention.kernels.gdn.mtp_fused_recurrent import mtp_fused_recurrent_gated_delta_rule

            mixed_qkv = _mtp_conv(config, inputs, att_state)
            query, key, value = _split_qkv(config, mixed_qkv)
            rows = query.shape[0]
            # num_accepted_tokens >= 1 is guaranteed upstream: cache restore sets 1 and MTP decode
            # only writes values in [1, draft_step + 1].
            values = value.view(1, rows, config.num_v_heads, config.head_v_dim)
            mtp_fused_recurrent_gated_delta_rule(
                q=query.view(1, rows, config.num_k_heads, config.head_k_dim),
                k=key.view(1, rows, config.num_k_heads, config.head_k_dim),
                v=values,
                initial_state=inputs.ssm_states,
                cu_seqlens=att_state.cu_seqlens.to(torch.long),
                ssm_state_indices=att_state.ssm_idx,
                ssm_state_write_indices=att_state.ssm_idx,
                num_accepted_tokens=att_state.num_accepted_tokens,
                A_log=inputs.A_log,
                dt_bias=inputs.dt_bias,
                a_raw=inputs.a,
                b_raw=inputs.b,
                out=out.view(values.shape),
            )
        return out


class ReplayGDNDecodeBackend(TritonGDNDecodeBackend):
    name = "replay_gdn_decode"
    replay = True

    def forward(
        self,
        att_state: GDNDecodeAttState,
        layer_inputs: GDNLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [rows, num_v_heads, head_v_dim]; update the conv state and append history.

        Every layer reads the cursors of the step's start; the last layer's call advances them.
        """
        from lightllm.common.attention.kernels.gdn.replayssm.replay_gdn import replay

        config, inputs = self.layer_config, layer_inputs
        assert inputs.replay_state is not None and inputs.layer_num is not None, "replay decode needs replay_state"
        if att_state.draft_step == 0:
            query, key, value, a, b = _pack_decode_inputs(config, inputs, att_state.conv_idx)
        else:
            query, key, value = _split_qkv(config, _mtp_conv(config, inputs, att_state))
            a, b = inputs.a, inputs.b
        query = query.view(-1, config.num_k_heads, config.head_k_dim)
        key = key.view(-1, config.num_k_heads, config.head_k_dim)
        value = value.view(-1, config.num_v_heads, config.head_v_dim)

        state = inputs.replay_state
        replay(
            state,
            state.storage_layer(inputs.layer_num),
            query,
            key,
            value,
            a,
            b,
            inputs.A_log,
            inputs.dt_bias,
            att_state.conv_idx,
            att_state.cu_seqlens,
            out=out.view(value.shape),
        )
        state.finish_decode_token(inputs.layer_num, att_state.conv_idx, att_state.cu_seqlens)
        return out

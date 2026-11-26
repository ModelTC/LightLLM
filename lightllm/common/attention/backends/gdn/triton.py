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

        cfg, inputs = self.layer_config, layer_inputs
        g, beta = fused_gdn_gating(inputs.A_log, inputs.a, inputs.b, inputs.dt_bias)
        conv_states = inputs.conv_states
        if cfg.state_mtp_step > 0:
            conv_states = conv_states[:, :, : -cfg.state_mtp_step]
        mixed_qkv = causal_conv1d_fn(
            inputs.mixed_qkv.transpose(0, 1),
            inputs.conv_weight,
            bias=inputs.conv_bias,
            query_start_loc=att_state.cu_seqlens,
            cache_indices=att_state.conv_idx,
            has_initial_state=att_state.has_initial_state,
            conv_states=conv_states,
            activation=cfg.activation,
        ).transpose(0, 1)
        query, key, value = _split_qkv(cfg, mixed_qkv)
        tokens = query.shape[0]
        ssm_states = inputs.ssm_states
        core, last_state = self.chunk_gated_delta_rule(
            q=query.view(1, tokens, cfg.num_k_heads, cfg.head_k_dim),
            k=key.view(1, tokens, cfg.num_k_heads, cfg.head_k_dim),
            v=value.view(1, tokens, cfg.num_v_heads, cfg.head_v_dim),
            g=g.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=ssm_states[att_state.ssm_idx],
            output_final_state=True,
            cu_seqlens=att_state.cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        # The chunk kernel accumulates in float32; index_put does not convert dtypes implicitly.
        ssm_states[att_state.ssm_idx] = last_state.to(ssm_states.dtype, copy=False)
        output = core.view(tokens, cfg.num_v_heads, cfg.head_v_dim)
        if out is None:
            return output
        out.copy_(output)
        return out

    def build(self, infer_state: InferStateInfo) -> GDNPrefillAttState:
        req, c = infer_state.b_req_idx, self.layer_config
        ssm_slots_per_request = 1 if c.replay else c.state_mtp_step + 1
        return GDNPrefillAttState(
            conv_idx=req,
            ssm_idx=req * ssm_slots_per_request if ssm_slots_per_request != 1 else req,
            cu_seqlens=infer_state.b1_cu_q_seq_len,
            has_initial_state=infer_state.b_ready_cache_len > 0,
        )


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
    req, device = infer_state.b_req_idx, infer_state.b_req_idx.device
    width = infer_state.decode_rows_per_request
    derived = {}
    if width == 1:
        conv_idx = ssm_idx = req
        accepted = None
        cu = None
        if replay:
            derived["cu"] = cu = torch.arange(req.numel() + 1, dtype=torch.int32, device=device)
    else:
        assert infer_state.req_to_mtp_state_index is not None, "speculative GDN decode needs req_to_mtp_state_index"
        if infer_state.dynamic_decode_rows:
            cu, conv_idx, accepted = build_dynamic_mtp_linear_att_state_params(
                b_req_idx=req,
                b_mtp_index=infer_state.b_mtp_index,
                req_to_mtp_state_index=infer_state.req_to_mtp_state_index,
                hold_req_id=infer_state.pad_req_idx,
            )
        else:
            assert infer_state.batch_size % width == 0, (
                "GDN fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
                f"got batch_size={infer_state.batch_size}, draft_step={width - 1}."
            )
            cu = torch.arange(0, infer_state.batch_size + 1, width, dtype=torch.int32, device=device)
            conv_idx = req.view(-1, width)[:, 0].contiguous()
            accepted = infer_state.req_to_mtp_state_index[conv_idx] + 1
        if replay:
            ssm_idx = conv_idx
        else:
            # Each request owns width consecutive recurrent-state slots.
            offsets = torch.arange(width, device=device, dtype=req.dtype).view(1, width)
            ssm_idx = (conv_idx * width).view(-1, 1) + offsets
        derived.update(cu=cu, conv_idx=conv_idx, accepted=accepted, ssm_idx=ssm_idx)
    if infer_state.uses_cuda_graph and derived:
        kept = {name: backend._keep_for_graph(infer_state, name, value) for name, value in derived.items()}
        cu = kept.get("cu", cu)
        conv_idx = kept.get("conv_idx", conv_idx)
        accepted = kept.get("accepted", accepted)
        ssm_idx = kept["conv_idx"] if replay and "conv_idx" in kept else kept.get("ssm_idx", ssm_idx)
    return GDNDecodeAttState(
        conv_idx=conv_idx, ssm_idx=ssm_idx, cu_seqlens=cu, num_accepted_tokens=accepted, draft_step=width - 1
    )


def _pack_decode_inputs(cfg: GDNLayerConfig, inputs: GDNLayerInputs, conv_idx: torch.Tensor):
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
        cfg.activation,
        cfg.conv_kernel_dim,
        cfg.num_k_heads,
        cfg.head_k_dim,
        cfg.num_v_heads,
        cfg.head_v_dim,
    )


def _mtp_conv(cfg: GDNLayerConfig, inputs: GDNLayerInputs, att_state: GDNDecodeAttState):
    from lightllm.common.attention.kernels.gdn.causal_conv1d_mtp import causal_conv1d_update

    return causal_conv1d_update(
        inputs.mixed_qkv,
        inputs.conv_states,
        inputs.conv_weight,
        mtp_step=att_state.draft_step,
        bias=inputs.conv_bias,
        activation=cfg.activation,
        conv_state_indices=att_state.conv_idx,
        num_accepted_tokens=att_state.num_accepted_tokens,
        query_start_loc=att_state.cu_seqlens,
    )


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

    def forward(
        self,
        att_state: GDNDecodeAttState,
        layer_inputs: GDNLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [rows, num_v_heads, head_v_dim]; update the conv and SSM states in place."""
        cfg, inputs = self.layer_config, layer_inputs
        if att_state.draft_step == 0:
            from lightllm.common.attention.kernels.gdn.third_party_fla.ops.fused_recurrent import (
                fused_recurrent_gated_delta_rule,
            )

            query, key, value, a, b = _pack_decode_inputs(cfg, inputs, att_state.conv_idx)
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

            mixed_qkv = _mtp_conv(cfg, inputs, att_state)
            query, key, value = _split_qkv(cfg, mixed_qkv)
            rows = query.shape[0]
            # num_accepted_tokens >= 1 is guaranteed upstream: cache restore sets 1 and MTP decode
            # only writes values in [1, draft_step + 1].
            values = value.view(1, rows, cfg.num_v_heads, cfg.head_v_dim)
            mtp_fused_recurrent_gated_delta_rule(
                q=query.view(1, rows, cfg.num_k_heads, cfg.head_k_dim),
                k=key.view(1, rows, cfg.num_k_heads, cfg.head_k_dim),
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

    def build(self, infer_state: InferStateInfo) -> GDNDecodeAttState:
        return build_recurrent_decode_state(self, infer_state)


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

        cfg, inputs = self.layer_config, layer_inputs
        assert inputs.replay_state is not None and inputs.layer_num is not None, "replay decode needs replay_state"
        if att_state.draft_step == 0:
            query, key, value, a, b = _pack_decode_inputs(cfg, inputs, att_state.conv_idx)
            query = query.view(query.shape[0], cfg.num_k_heads, cfg.head_k_dim)
            key = key.view(key.shape[0], cfg.num_k_heads, cfg.head_k_dim)
            value = value.view(value.shape[0], cfg.num_v_heads, cfg.head_v_dim)
        else:
            query, key, value = _split_qkv(cfg, _mtp_conv(cfg, inputs, att_state))
            query = query.view(-1, cfg.num_k_heads, cfg.head_k_dim)
            key = key.view(-1, cfg.num_k_heads, cfg.head_k_dim)
            value = value.view(-1, cfg.num_v_heads, cfg.head_v_dim)
            a, b = inputs.a, inputs.b
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

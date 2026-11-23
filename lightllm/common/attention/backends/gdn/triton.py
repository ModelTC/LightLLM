"""GDN step builders and the Triton (in-tree FLA) implementations over native recurrent state."""

from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionImpl,
    AttentionStepBuilder,
    StepBuilderContext,
    TensorAllocator,
)
from .. import keep_for_cuda_graph
from . import GDNLayerConfig, GDNLayerInputs


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


class GDNPrefillStepBuilder(AttentionStepBuilder[GDNPrefillAttState]):
    def __init__(self, *, ssm_slots_per_request: int) -> None:
        """Native state keeps state_mtp_step + 1 SSM slots per request; replay state keeps one."""
        self.ssm_slots_per_request = ssm_slots_per_request

    def build(self, batch: AttentionMetadata) -> GDNPrefillAttState:
        req = batch.b_req_idx
        return GDNPrefillAttState(
            conv_idx=req,
            ssm_idx=req * self.ssm_slots_per_request if self.ssm_slots_per_request != 1 else req,
            cu_seqlens=batch.cu_q_seq_len,
            has_initial_state=batch.b_ready_cache_len > 0,
        )


class GDNDecodeStepBuilder(AttentionStepBuilder[GDNDecodeAttState]):
    def __init__(self, *, replay: bool) -> None:
        """replay selects replay state, which keeps one SSM slot per request and needs cu_seqlens."""
        self.replay = replay
        self._cuda_graph_tensors: dict[tuple, torch.Tensor] = {}

    def build(self, batch: AttentionMetadata) -> GDNDecodeAttState:
        """A CUDA graph step keeps derived tensors per (microbatch, batch size), rewritten in place."""
        from lightllm.common.attention.kernels.gdn.mtp_state_params import build_dynamic_mtp_linear_att_state_params

        req, device = batch.b_req_idx, batch.b_req_idx.device
        width = batch.decode_rows_per_request
        derived = {}
        if width == 1:
            conv_idx = ssm_idx = req
            accepted = None
            cu = None
            if self.replay:
                derived["cu"] = cu = torch.arange(req.numel() + 1, dtype=torch.int32, device=device)
        else:
            assert batch.req_to_mtp_state_index is not None, "speculative GDN decode needs req_to_mtp_state_index"
            if batch.dynamic_decode_rows:
                cu, conv_idx, accepted = build_dynamic_mtp_linear_att_state_params(
                    b_req_idx=req,
                    b_mtp_index=batch.b_mtp_index,
                    req_to_mtp_state_index=batch.req_to_mtp_state_index,
                    hold_req_id=batch.pad_req_idx,
                )
            else:
                assert batch.batch_size % width == 0, (
                    "GDN fixed-layout decode requires batch_size to be divisible by draft_step + 1, "
                    f"got batch_size={batch.batch_size}, draft_step={width - 1}."
                )
                cu = torch.arange(0, batch.batch_size + 1, width, dtype=torch.int32, device=device)
                conv_idx = req.view(-1, width)[:, 0].contiguous()
                accepted = batch.req_to_mtp_state_index[conv_idx] + 1
            if self.replay:
                ssm_idx = conv_idx
            else:
                # Each request owns width consecutive recurrent-state slots.
                offsets = torch.arange(width, device=device, dtype=req.dtype).view(1, width)
                ssm_idx = (conv_idx * width).view(-1, 1) + offsets
            derived.update(cu=cu, conv_idx=conv_idx, accepted=accepted, ssm_idx=ssm_idx)
        if batch.uses_cuda_graph and derived:
            key, store = (batch.microbatch_index, batch.batch_size), self._cuda_graph_tensors
            kept = {name: keep_for_cuda_graph(store, (*key, name), value) for name, value in derived.items()}
            cu = kept.get("cu", cu)
            conv_idx = kept.get("conv_idx", conv_idx)
            accepted = kept.get("accepted", accepted)
            ssm_idx = kept["conv_idx"] if self.replay and "conv_idx" in kept else kept.get("ssm_idx", ssm_idx)
        return GDNDecodeAttState(
            conv_idx=conv_idx, ssm_idx=ssm_idx, cu_seqlens=cu, num_accepted_tokens=accepted, draft_step=width - 1
        )


def _split_qkv(layer_config: GDNLayerConfig, mixed_qkv: torch.Tensor):
    key_dim = layer_config.num_k_heads * layer_config.head_k_dim
    value_dim = layer_config.num_v_heads * layer_config.head_v_dim
    return torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)


def _prefill_state_views(layer_config: GDNLayerConfig, layer_inputs: GDNLayerInputs):
    conv_states = layer_inputs.conv_states
    # Speculative conv columns past the kernel width are not used by prefill.
    if layer_config.state_mtp_step > 0:
        conv_states = conv_states[:, :, : -layer_config.state_mtp_step]
    return conv_states


class TritonGDNPrefillImpl(AttentionImpl[GDNLayerConfig, GDNPrefillAttState, GDNLayerInputs]):
    """Chunked prefill with the in-tree FLA kernel; subclasses swap chunk_gated_delta_rule."""

    def __init__(self, layer_config: GDNLayerConfig) -> None:
        super().__init__(layer_config)
        # Resolve the chunk kernel now, so a missing or broken optional library is rejected when the
        # model loads (auto-selection then moves on) rather than failing a step.
        try:
            self.chunk_gated_delta_rule()
        except Exception as e:
            raise NotImplementedError(f"{type(self).__name__} cannot load its chunk kernel: {e!r}") from e

    @staticmethod
    def chunk_gated_delta_rule():
        from lightllm.common.attention.kernels.gdn.third_party_fla.ops.chunk import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    def forward(
        self,
        att_state: GDNPrefillAttState,
        layer_inputs: GDNLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [tokens, num_v_heads, head_v_dim]; update the conv and SSM states in place."""
        from lightllm.common.attention.kernels.gdn.causal_conv1d import causal_conv1d_fn
        from lightllm.common.attention.kernels.gdn.fused_gdn_gating import fused_gdn_gating

        cfg, inputs = self.layer_config, layer_inputs
        g, beta = fused_gdn_gating(inputs.A_log, inputs.a, inputs.b, inputs.dt_bias)
        mixed_qkv = causal_conv1d_fn(
            inputs.mixed_qkv.transpose(0, 1),
            inputs.conv_weight,
            bias=inputs.conv_bias,
            query_start_loc=att_state.cu_seqlens,
            cache_indices=att_state.conv_idx,
            has_initial_state=att_state.has_initial_state,
            conv_states=_prefill_state_views(cfg, inputs),
            activation=cfg.activation,
        ).transpose(0, 1)
        query, key, value = _split_qkv(cfg, mixed_qkv)
        tokens = query.shape[0]
        ssm_states = inputs.ssm_states
        core, last_state = self.chunk_gated_delta_rule()(
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
        out.copy_(core.view(out.shape))
        return out


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


class TritonGDNDecodeImpl(AttentionImpl[GDNLayerConfig, GDNDecodeAttState, GDNLayerInputs]):
    def __init__(self, layer_config: GDNLayerConfig) -> None:
        super().__init__(layer_config)
        if layer_config.replay:
            raise NotImplementedError("Triton GDN decode updates native SSM states; replay state needs replay decode")

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
            core, _ = fused_recurrent_gated_delta_rule(
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
            )
        else:
            from lightllm.common.attention.kernels.gdn.mtp_fused_recurrent import mtp_fused_recurrent_gated_delta_rule

            mixed_qkv = _mtp_conv(cfg, inputs, att_state)
            query, key, value = _split_qkv(cfg, mixed_qkv)
            rows = query.shape[0]
            # num_accepted_tokens >= 1 is guaranteed upstream: cache restore sets 1 and MTP decode
            # only writes values in [1, draft_step + 1].
            core, _ = mtp_fused_recurrent_gated_delta_rule(
                q=query.view(1, rows, cfg.num_k_heads, cfg.head_k_dim),
                k=key.view(1, rows, cfg.num_k_heads, cfg.head_k_dim),
                v=value.view(1, rows, cfg.num_v_heads, cfg.head_v_dim),
                initial_state=inputs.ssm_states,
                cu_seqlens=att_state.cu_seqlens.to(torch.long),
                ssm_state_indices=att_state.ssm_idx,
                ssm_state_write_indices=att_state.ssm_idx,
                num_accepted_tokens=att_state.num_accepted_tokens,
                A_log=inputs.A_log,
                dt_bias=inputs.dt_bias,
                a_raw=inputs.a,
                b_raw=inputs.b,
            )
        out.copy_(core.view(out.shape))
        return out


class TritonGDNPrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "triton_gdn_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[GDNPrefillStepBuilder]:
        return GDNPrefillStepBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonGDNPrefillImpl]:
        return TritonGDNPrefillImpl

    @classmethod
    def create_step_builder(cls, layer_config: GDNLayerConfig, ctx: StepBuilderContext) -> GDNPrefillStepBuilder:
        return create_gdn_prefill_step_builder(layer_config)

    @classmethod
    def step_builder_key(cls, layer_config: GDNLayerConfig):
        return (layer_config.replay, layer_config.state_mtp_step)


class TritonGDNDecodeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "triton_gdn_decode"

    @staticmethod
    def get_step_builder_cls() -> type[GDNDecodeStepBuilder]:
        return GDNDecodeStepBuilder

    @staticmethod
    def get_impl_cls() -> type[TritonGDNDecodeImpl]:
        return TritonGDNDecodeImpl

    @classmethod
    def create_step_builder(cls, layer_config: GDNLayerConfig, ctx: StepBuilderContext) -> GDNDecodeStepBuilder:
        return GDNDecodeStepBuilder(replay=layer_config.replay)

    @classmethod
    def step_builder_key(cls, layer_config: GDNLayerConfig):
        return layer_config.replay


def create_gdn_prefill_step_builder(layer_config) -> GDNPrefillStepBuilder:
    """The prefill step builder of GDN or KDA layers: native state keeps state_mtp_step + 1 SSM slots per request."""
    return GDNPrefillStepBuilder(ssm_slots_per_request=1 if layer_config.replay else layer_config.state_mtp_step + 1)

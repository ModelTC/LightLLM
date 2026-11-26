"""Triton GDN implementations and decode state preparation."""

from dataclasses import dataclass
from typing import Optional

import torch

from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionBuildResources,
    TensorAllocator,
)
from lightllm.common.basemodel.infer_struct import InferStateInfo
from .. import keep_for_cuda_graph
from . import GDNLayerConfig, GDNLayerInputs, _split_qkv
from .prefill import GDNPrefillBackend


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


class GDNDecodeBuildContext:
    def __init__(self, *, replay: bool) -> None:
        """replay selects replay state, which keeps one SSM slot per request and needs cu_seqlens."""
        self.replay = replay
        self._cuda_graph_tensors: dict[tuple, torch.Tensor] = {}

    def prepare(self, infer_state: InferStateInfo) -> GDNDecodeAttState:
        """A CUDA graph step keeps derived tensors per (microbatch, batch size), rewritten in place."""
        from lightllm.common.attention.kernels.gdn.mtp_state_params import build_dynamic_mtp_linear_att_state_params

        req, device = infer_state.b_req_idx, infer_state.b_req_idx.device
        width = infer_state.decode_rows_per_request
        derived = {}
        if width == 1:
            conv_idx = ssm_idx = req
            accepted = None
            cu = None
            if self.replay:
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
            if self.replay:
                ssm_idx = conv_idx
            else:
                # Each request owns width consecutive recurrent-state slots.
                offsets = torch.arange(width, device=device, dtype=req.dtype).view(1, width)
                ssm_idx = (conv_idx * width).view(-1, 1) + offsets
            derived.update(cu=cu, conv_idx=conv_idx, accepted=accepted, ssm_idx=ssm_idx)
        if infer_state.uses_cuda_graph and derived:
            key, store = (infer_state.microbatch_index, infer_state.batch_size), self._cuda_graph_tensors
            kept = {name: keep_for_cuda_graph(store, (*key, name), value) for name, value in derived.items()}
            cu = kept.get("cu", cu)
            conv_idx = kept.get("conv_idx", conv_idx)
            accepted = kept.get("accepted", accepted)
            ssm_idx = kept["conv_idx"] if self.replay and "conv_idx" in kept else kept.get("ssm_idx", ssm_idx)
        return GDNDecodeAttState(
            conv_idx=conv_idx, ssm_idx=ssm_idx, cu_seqlens=cu, num_accepted_tokens=accepted, draft_step=width - 1
        )


class TritonGDNPrefillBackend(GDNPrefillBackend):
    @staticmethod
    def chunk_gated_delta_rule():
        from lightllm.common.attention.kernels.gdn.third_party_fla.ops.chunk import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    @staticmethod
    def get_name() -> str:
        return "triton_gdn_prefill"


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
    @classmethod
    def _check_supported(cls, layer_config: GDNLayerConfig) -> None:
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

    @staticmethod
    def get_name() -> str:
        return "triton_gdn_decode"

    build_context_type = GDNDecodeBuildContext

    @classmethod
    def build(cls, metadata: InferStateInfo, context: GDNDecodeBuildContext) -> GDNDecodeAttState:
        return context.prepare(metadata)

    @classmethod
    def create_build_context(cls, build_key: bool, ctx: AttentionBuildResources) -> GDNDecodeBuildContext:
        return GDNDecodeBuildContext(replay=build_key)

    @classmethod
    def build_context_key(cls, layer_config: GDNLayerConfig):
        return layer_config.replay

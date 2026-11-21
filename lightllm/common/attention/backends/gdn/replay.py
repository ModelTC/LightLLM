"""GDN decode over replay state: instead of updating SSM states every token, rows append to a per-layer
history that the replay kernel reads from each request's cursor. Prefill is the regular chunked prefill
with one SSM slot per request."""

import torch

from lightllm.common.attention.backend import AttentionBackend, AttentionImpl, TensorAllocator
from . import GDNLayerConfig, GDNLayerInputs
from .triton import GDNDecodeStepArgs, GDNDecodeStepBuilder, _mtp_conv, _pack_decode_inputs, _split_qkv


class ReplayGDNDecodeImpl(AttentionImpl[GDNLayerConfig, GDNDecodeStepArgs, GDNLayerInputs]):
    def forward(
        self,
        step_args: GDNDecodeStepArgs,
        layer_inputs: GDNLayerInputs,
        *,
        out: torch.Tensor,
        alloc_func: TensorAllocator = torch.empty,
    ) -> torch.Tensor:
        """Fill and return out [rows, num_v_heads, head_v_dim]; update the conv state and append history.

        The caller advances the replay cursors after the step's last layer.
        """
        from lightllm.common.attention.kernels.gdn.replayssm.replay_gdn import replay

        cfg, inputs = self.layer_config, layer_inputs
        assert inputs.replay_state is not None and inputs.layer_num is not None, "replay decode needs replay_state"
        if step_args.draft_step == 0:
            query, key, value, a, b = _pack_decode_inputs(cfg, inputs, step_args.conv_idx)
            query = query.view(query.shape[0], cfg.num_k_heads, cfg.head_k_dim)
            key = key.view(key.shape[0], cfg.num_k_heads, cfg.head_k_dim)
            value = value.view(value.shape[0], cfg.num_v_heads, cfg.head_v_dim)
        else:
            query, key, value = _split_qkv(cfg, _mtp_conv(cfg, inputs, step_args))
            query = query.view(-1, cfg.num_k_heads, cfg.head_k_dim)
            key = key.view(-1, cfg.num_k_heads, cfg.head_k_dim)
            value = value.view(-1, cfg.num_v_heads, cfg.head_v_dim)
            a, b = inputs.a, inputs.b
        state = inputs.replay_state
        core = replay(
            state,
            state.storage_layer(inputs.layer_num),
            query,
            key,
            value,
            a,
            b,
            inputs.A_log,
            inputs.dt_bias,
            step_args.conv_idx,
            step_args.cu_seqlens,
        )
        out.copy_(core.view(out.shape))
        return out


class ReplayGDNDecodeBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "replay_gdn_decode"

    @staticmethod
    def get_step_builder_cls() -> type[GDNDecodeStepBuilder]:
        return GDNDecodeStepBuilder

    @staticmethod
    def get_impl_cls() -> type[ReplayGDNDecodeImpl]:
        return ReplayGDNDecodeImpl

"""GDN decode over replay state: instead of updating SSM states every token, rows append to a per-layer
history that the replay kernel reads from each request's cursor. Prefill is the regular chunked prefill
with one SSM slot per request."""

import torch

from lightllm.common.attention.backend import AttentionBackend, AttentionImpl, StepBuilderContext, TensorAllocator
from . import GDNLayerConfig, GDNLayerInputs
from .triton import GDNDecodeStepArgs, GDNDecodeStepBuilder, _mtp_conv, _pack_decode_inputs, _split_qkv


class ReplayGDNDecodeImpl(AttentionImpl[GDNLayerConfig, GDNDecodeStepArgs, GDNLayerInputs]):
    def __init__(self, layer_config: GDNLayerConfig) -> None:
        super().__init__(layer_config)
        if not layer_config.replay:
            raise NotImplementedError("replay GDN decode needs replay state")

    def forward(
        self,
        step_args: GDNDecodeStepArgs,
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
        state.finish_decode_token(inputs.layer_num, step_args.conv_idx, step_args.cu_seqlens)
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

    @classmethod
    def create_step_builder(cls, layer_config: GDNLayerConfig, ctx: StepBuilderContext) -> GDNDecodeStepBuilder:
        return GDNDecodeStepBuilder(replay=True)

    @classmethod
    def step_builder_key(cls, layer_config: GDNLayerConfig):
        return True

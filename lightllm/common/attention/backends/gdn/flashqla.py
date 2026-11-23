"""GDN prefill with the FlashQLA chunk kernel; decode uses the Triton implementations."""

from lightllm.common.attention.backend import AttentionBackend, StepBuilderContext
from . import GDNLayerConfig
from .triton import GDNPrefillStepBuilder, TritonGDNPrefillImpl, create_gdn_prefill_step_builder


class FlashQLAGDNPrefillImpl(TritonGDNPrefillImpl):
    @staticmethod
    def chunk_gated_delta_rule():
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule


class FlashQLAGDNPrefillBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "flashqla_gdn_prefill"

    @staticmethod
    def get_step_builder_cls() -> type[GDNPrefillStepBuilder]:
        return GDNPrefillStepBuilder

    @staticmethod
    def get_impl_cls() -> type[FlashQLAGDNPrefillImpl]:
        return FlashQLAGDNPrefillImpl

    @classmethod
    def create_step_builder(cls, layer_config: GDNLayerConfig, ctx: StepBuilderContext) -> GDNPrefillStepBuilder:
        return create_gdn_prefill_step_builder(layer_config)

    @classmethod
    def step_builder_key(cls, layer_config: GDNLayerConfig):
        return (layer_config.replay, layer_config.state_mtp_step)

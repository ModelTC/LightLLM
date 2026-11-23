"""GDN prefill with the FlashQLA chunk kernel; decode uses the Triton implementations."""

from lightllm.common.attention.backend import AttentionBackend, AttentionBuildResources
from . import GDNLayerConfig
from .triton import GDNPrefillBuildContext, TritonGDNPrefillBackend, create_gdn_prefill_step_builder


class FlashQLAGDNPrefillBackend(TritonGDNPrefillBackend):
    @staticmethod
    def chunk_gated_delta_rule():
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    @staticmethod
    def get_name() -> str:
        return "flashqla_gdn_prefill"

    build_context_type = GDNPrefillBuildContext

    @classmethod
    def create_build_context(cls, layer_config: GDNLayerConfig, ctx: AttentionBuildResources) -> GDNPrefillBuildContext:
        return create_gdn_prefill_step_builder(layer_config)

    @classmethod
    def build_context_key(cls, layer_config: GDNLayerConfig):
        return (layer_config.replay, layer_config.state_mtp_step)

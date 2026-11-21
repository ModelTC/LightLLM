"""GDN prefill with the FlashQLA chunk kernel; decode uses the Triton implementations."""

from lightllm.common.attention.backend import AttentionBackend
from .triton import GDNPrefillStepBuilder, TritonGDNPrefillImpl


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

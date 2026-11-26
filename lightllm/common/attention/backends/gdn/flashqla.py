"""GDN prefill with the FlashQLA chunk kernel; decode uses the Triton implementations."""

from .prefill import GDNPrefillBackend


class FlashQLAGDNPrefillBackend(GDNPrefillBackend):
    @staticmethod
    def chunk_gated_delta_rule():
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    @staticmethod
    def get_name() -> str:
        return "flashqla_gdn_prefill"

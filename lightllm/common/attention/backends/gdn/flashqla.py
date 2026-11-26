"""GDN prefill with the FlashQLA chunk kernel; decode uses the Triton implementations."""

from .triton import TritonGDNPrefillBackend


class FlashQLAGDNPrefillBackend(TritonGDNPrefillBackend):
    name = "flashqla_gdn_prefill"
    phase = "prefill"

    @staticmethod
    def load_chunk_kernel():
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule

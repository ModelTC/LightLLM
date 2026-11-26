"""使用 FlashQLA 分块内核进行 GDN prefill；解码使用 Triton 实现。"""

from .triton import TritonGDNPrefillBackend


class FlashQLAGDNPrefillBackend(TritonGDNPrefillBackend):
    @staticmethod
    def chunk_gated_delta_rule():
        from flash_qla import chunk_gated_delta_rule

        return chunk_gated_delta_rule

    @staticmethod
    def get_name() -> str:
        return "flashqla_gdn_prefill"

from lightllm.common.attention.backends.gdn.flashqla import FlashQLAGDNPrefillBackend
from lightllm.common.basemodel.attention.linear.gdn import LinearAttBackend


class FlashQlaLinearAttBackend(LinearAttBackend):
    """Native GDN state, FlashQLA chunked prefill."""

    prefill_backend = FlashQLAGDNPrefillBackend

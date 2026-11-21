"""GDN over replay state: prefill is the regular chunked prefill with one SSM slot per request; decode
appends history that the replay kernel reads from each request's cursor."""

from lightllm.common.attention.backends.gdn.flashqla import FlashQLAGDNPrefillBackend
from lightllm.common.attention.backends.gdn.replay import ReplayGDNDecodeBackend
from .gdn import LinearAttBackend


class ReplayGDNTritonAttBackend(LinearAttBackend):
    """Replay GDN state, Triton chunked prefill."""

    decode_backend = ReplayGDNDecodeBackend
    replay = True


class ReplayGDNFlashQlaAttBackend(LinearAttBackend):
    """Replay GDN state, FlashQLA chunked prefill."""

    prefill_backend = FlashQLAGDNPrefillBackend
    decode_backend = ReplayGDNDecodeBackend
    replay = True

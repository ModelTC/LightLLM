from lightllm.utils.neo_fa3_utils import HAS_FLASH_ATTN_INTERFACE, flash_attn_with_kvcache_neo  # noqa: F401
from .fp import Fa3AttBackend


class NeoFa3AttBackend(Fa3AttBackend):
    """FA3 prefill for Neo models, whose layers attend bidirectionally inside image spans.

    The layers ask for it with AttControl.image_spans; FA3 runs it with the Neo flash_attn_interface
    build (see lightllm.utils.neo_fa3_utils).
    """

from .int8kv import Int8kvTritonAttBackend, QuantTritonDecodeAttState, QuantTritonPrefillAttState


class Int4kvTritonAttBackend(Int8kvTritonAttBackend):
    """Triton attention over a group-quantized int4 KV cache (two values per int8 byte)."""

    kv_quant: str = "int4_group"

    def _uses_diverse_decode(self) -> bool:
        return False


# Names kept for code that refers to the int4 states.
Int4kvTritonPrefillAttState = QuantTritonPrefillAttState
Int4kvTritonDecodeAttState = QuantTritonDecodeAttState
